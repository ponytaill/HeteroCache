# Standard library imports
import math
import time
import warnings

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union

# Flash Attention imports
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

# Transformers imports
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.modeling_flash_attention_utils import _upad_input
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.utils.deprecation import deprecate_kwarg

# Local imports
from heterocache.cache_utils import HeteroCache
from tools.log import get_logger

logger = get_logger()


def llama_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
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
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
def llama_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        # `position_embeddings` is deprecated and will be removed in v4.46
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(query_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
                "num_key_value_groups": self.num_key_value_groups,
                "query_states": query_states,
                "update_global_past_kv": getattr(self, "update_global_past_kv", True),
            }
            (
                key_states,
                value_states,
            ) = past_key_value.update(  # DynamicCache/KvcompressCache
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        
        # GQA/MQA: Repeat K/V heads to match Q heads
        if query_states.shape[1] != key_states.shape[1]:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Chunked attention for memory efficiency
        chunk_size = max(1, int(q_len / 100))
        if q_len > 1:
            # Prefill: chunked attention to save memory
            attn_output = torch.zeros_like(query_states)

            for i in range(0, q_len, chunk_size):
                chunk_end = min(i + chunk_size, q_len)
                query_chunk = query_states[:, :, i:chunk_end, :]
                attn_weights_chunk = torch.matmul(query_chunk, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_mask is not None:
                    causal_mask_chunk = attention_mask[:, :, i:chunk_end, : key_states.shape[-2]]
                    attn_weights_chunk = attn_weights_chunk + causal_mask_chunk

                attn_weights_chunk = nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_chunk = nn.functional.dropout(attn_weights_chunk, p=self.attention_dropout, training=self.training)
                attn_output_chunk = torch.matmul(attn_weights_chunk, value_states)
                attn_output[:, :, i:chunk_end, :] = attn_output_chunk

            attn_weights = None
        else:
            # Decode: standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
def llama_rms_forward(self, hidden_states):
        input_dtype = hidden_states.dtype

        if input_dtype == torch.float32:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states

        # Chunked computation for memory efficiency
        chunk_size = 1024
        chunks = hidden_states.split(chunk_size, dim=1)

        processed_chunks = []
        for chunk in chunks:
            chunk_fp32 = chunk.to(torch.float32)
            variance = chunk_fp32.pow(2).mean(-1, keepdim=True)
            normed_chunk = chunk_fp32 * torch.rsqrt(variance + self.variance_epsilon)
            processed_chunks.append(normed_chunk.to(input_dtype))

        hidden_states = torch.cat(processed_chunks, dim=1)
        return self.weight * hidden_states
def llama_mlp_forward(self, x):
    if self.config.pretraining_tp > 1:
        slice = self.intermediate_size // self.config.pretraining_tp
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        # Chunked MLP computation for memory efficiency
        chunk_size = 512
        x_chunks = x.split(chunk_size, dim=1)
        down_proj_chunks = []

        for x_chunk in x_chunks:
            gate_chunk = self.gate_proj(x_chunk)
            up_chunk = self.up_proj(x_chunk)
            intermediate_chunk = self.act_fn(gate_chunk) * up_chunk
            down_chunk = self.down_proj(intermediate_chunk)
            down_proj_chunks.append(down_chunk)
            del gate_chunk, up_chunk, intermediate_chunk

        down_proj = torch.cat(down_proj_chunks, dim=1)

    return down_proj


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        past_key_values: HeteroCache = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            if query_states.shape[-2] != 1:
                # Prefill stage
                past_key_values.init_coefficient(query_states, key_states, self.layer_idx)
                past_key_values.prefill_select(query_states, key_states, value_states, self.layer_idx)

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
                # Decode stage: Paged Attention
                block_table = past_key_values.block_tables[self.layer_idx]
                k_pool = past_key_values.key_pools[self.layer_idx]
                v_pool = past_key_values.value_pools[self.layer_idx]
                cache_seqlens = past_key_values.heads_len[self.layer_idx].view(-1).int()

                # Prepare query for GQA
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

                past_key_values.decode_select(query_states, self.layer_idx)

                # Restore output shape
                attn_output = attn_output.view(bsz, self.num_key_value_heads, q_len, group_size, self.head_dim)
                attn_output = attn_output.permute(0, 2, 1, 3, 4).reshape(bsz, q_len, self.num_heads, self.head_dim)
                attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                attn_output = self.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None

            

        
        return attn_output, attn_weights

last_call_t = time.time()


def time_analyze():
    """Calculate time elapsed since last call (in seconds)"""
    global last_call_t
    temp = round(time.time() - last_call_t, 4)
    last_call_t = time.time()
    return temp


def LlamaForCausalLM_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        n = input_ids.shape[1]
    
        if n > 1:
            logger.info(f"---prefill time {round(time_analyze(), 3)}s")
        else:
            logger.info(f"---decoding time {round(time_analyze(), 3)}s")
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )