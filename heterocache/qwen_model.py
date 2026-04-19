# Standard library imports
import time
import warnings
from typing import Callable, List, Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flash Attention imports
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

# Transformers imports
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.modeling_flash_attention_utils import _upad_input, FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward
)
from transformers.processing_utils import Unpack

# Local imports
from heterocache.cache_utils import HeteroCache
from tools.log import get_logger


def qwen2_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Qwen2 flash attention forward pass with optional KV cache and sliding window support."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "attention_mask": attention_mask,
            "num_key_value_groups": self.num_key_value_groups,
            "query_states": query_states,
            "update_global_past_kv": getattr(self, "update_global_past_kv", True),
        }
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with sliding window support."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool = False,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with KV cache support and paged attention for decode stage."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
                "num_key_value_groups": self.num_key_value_groups,
                "query_states": query_states,
                "update_global_past_kv": getattr(self, "update_global_past_kv", True),
            }
            key_states,value_states,= past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # Prefill stage: Q length > 1
        if query_states.shape[-2] != 1:
                past_key_values.init_coefficient(query_states,key_states,self.layer_idx)
                past_key_values.prefill_select(query_states, key_states, value_states,self.layer_idx)

                attn_output, attn_weights = flash_attention_forward(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                )
                assert attn_output.size(1) == q_len
                attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                attn_output = self.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None
        else:
            # Decode stage: Paged attention with Q length = 1
            past_key_values.decode_select(query_states, self.layer_idx)

            # Get paged KV cache metadata
            block_table = past_key_values.block_tables[self.layer_idx]
            k_pool = past_key_values.key_pools[self.layer_idx]
            v_pool = past_key_values.value_pools[self.layer_idx]

            cache_seqlens = past_key_values.heads_len[self.layer_idx].view(-1).int()

            # Prepare query for GQA: flatten batch dimension to [BSZ * KV_Heads]
            group_size = self.num_heads // self.num_key_value_heads
            q_in = query_states.view(bsz, self.num_key_value_heads, group_size, 1, self.head_dim)
            q_in = q_in.permute(0, 1, 3, 2, 4)
            q_in = q_in.reshape(-1, 1, group_size, self.head_dim)

            # Flash attention with paged KV cache
            attn_output = flash_attn_with_kvcache(
                q_in,
                k_pool,
                v_pool,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=False
            )

            # Restore output shape
            attn_output = attn_output.view(bsz, self.num_key_value_heads, q_len, group_size, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3, 4).reshape(bsz, q_len, self.num_heads, self.head_dim)

            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None
        return attn_output, attn_weights
