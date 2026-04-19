import os
import time
import transformers
from transformers.cache_utils import Cache, DynamicCache
import statistics
import copy

import torch
from typing import List, Optional, Dict, Tuple, Any,Type
import torch.nn.functional as F
import torch.nn as nn
import math
from heterocache.kv_cluster import *

def get_compress_len(compression_ratio, full_heads, ori_len, num_kv_heads):
    return int((compression_ratio * num_kv_heads - full_heads)* ori_len / (num_kv_heads - full_heads))

class CompressionCacheConfig():

    def __init__(
        self,
        window_size: Optional[int] = 32,
        # max_capacity_prompt: Optional[int] = 2048,
        compression_ratio: Optional[float] = 0.5,
        pooling: Optional[str] = "avgpool",
        kernel_size: Optional[int] = 5,
        num_attn_heads: Optional[int] = 32,
        num_kv_heads: Optional[int] = 8,
        num_layers: Optional[int] = 32,

    ):
        self.window_size = window_size
        self.compression_ratio = compression_ratio
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.num_attn_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.n_init = 128

class HeteroCacheConfig():
    def __init__(
        self,
        data: Optional[Dict] ,
    
        compression_ratio: Optional[float] = 0.5,
        stable_threshold: Optional[float] = 0.5,
        real_offload: Optional[bool] = True,
        max_gen_len: Optional[int] = 128,
        decode_step: Optional[int] = 4,
        pooling: Optional[str] = "avgpool",
        kernel_size: Optional[int] = 5,
        num_attn_heads: Optional[int] = 32,
        num_kv_heads: Optional[int] = 8,
        num_layers: Optional[int] = 32,
    ):
        self.data = data
        self.stable_threshold = stable_threshold 
        self.compression_ratio = compression_ratio
        self.real_offload = real_offload
        self.max_gen_len = max_gen_len
        self.decode_step = decode_step
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.quant_method = None
        self.num_attn_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class BaseKVCache(DynamicCache):
    def __init__(self):
        super().__init__()
        # Used in `generate` to keep tally of how many tokens the cache has seen
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.kv_clusters = {}
        self.kv_cluster_granularity = "layer"

        self.temp_key_cache = []
        self.temp_value_cache = []

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        raise NotImplementedError(
            "Make sure to implement `get_kv_cluster_class_config` in a subclass."
        )

    def get_kv_cluster_class(self, layer_idx: int, head_idx=None):
        cluster_name, cluster_class, cluster_config = self.get_kv_cluster_class_config(
            layer_idx, head_idx
        )
        if cluster_name not in self.kv_clusters:
            self.kv_clusters[cluster_name] = cluster_class(**cluster_config)
        return self.kv_clusters[cluster_name]

    def compressed_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        layer_idx: int,
    ):
        if self.kv_cluster_granularity == "layer":
            kv_cluster = self.get_kv_cluster_class(layer_idx)

            key_compress, value_compress = kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
            )
            self.key_cache.append(key_compress)
            self.value_cache.append(value_compress)
        else:
            assert (
                False
            ), f"kv_cluster_granularity {self.kv_cluster_granularity} not supported"

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        # if prefill, then compress; if decode, then update
        # [bsz, num_heads, q_len, head_dim]

        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        query_states = cache_kwargs["query_states"]
        attention_mask = cache_kwargs["attention_mask"]
        num_key_value_groups = cache_kwargs["num_key_value_groups"]

        if key_states.size(1) != query_states.size(1):  # GQA
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        q_len = query_states.shape[-2]
        initializing_kv_cluster = False
        if (
            len(self.key_cache) == layer_idx
        ):  # initialize kv_cluster, ie, the first query/context
            initializing_kv_cluster = True
            self.compressed_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
                layer_idx,
            )
        else:  # the follow up queries/contexts
            
            if update_global_past_kv:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            else:  # add KVs to temp_kv_cache
                if len(self.temp_key_cache) == layer_idx:
                    self.temp_key_cache.append(key_states)
                    self.temp_value_cache.append(value_states)
                else:
                    self.temp_key_cache[layer_idx] = torch.cat(
                        [self.temp_key_cache[layer_idx], key_states], dim=-2
                    )
                    self.temp_value_cache[layer_idx] = torch.cat(
                        [self.temp_value_cache[layer_idx], value_states], dim=-2
                    )

        # torch.cuda.empty_cache()
        
        if not initializing_kv_cluster:  # return the compressed KV cache
            if self.temp_key_cache:  # concat global past_kv and temp_kv_cache
                key_states = torch.cat(
                    [self.key_cache[layer_idx], self.temp_key_cache[layer_idx]], dim=-2
                )
                value_states = torch.cat(
                    [self.value_cache[layer_idx], self.temp_value_cache[layer_idx]],
                    dim=-2,
                )
            else:
                key_states = self.key_cache[layer_idx]
                value_states = self.value_cache[layer_idx]
        
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def to_legacy_cache(self):
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values):
        cache = cls()
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx]
            cache.update(key_states, value_states, layer_idx)
        return cache

    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[-1].shape[
                -2
            ]  # seq_len of temp_kv_cache
        self.temp_key_cache = []
        self.temp_value_cache = []


class H2OKVCache(BaseKVCache):
    def __init__(self, config):
        super().__init__()
        self.window_size = config.window_size
        # self.max_capacity_prompt = config.max_capacity_prompt
        self.compression_ratio = config.compression_ratio
        self.kernel_size = config.kernel_size
        self.pooling = config.pooling
   

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "window_size": self.window_size,
            "compression_ratio": self.compression_ratio,
            "kernel_size": self.kernel_size,
            "pooling": self.pooling,
        }
        cluster_name = ",".join(["h2o"] + [str(i) for i in cluster_config.values()])
        return cluster_name, H2OKVCluster, cluster_config
    
class SnapKVCache(BaseKVCache):
    def __init__(self, config):
        super().__init__()
        self.window_size = config.window_size
        # self.max_capacity_prompt = config.max_capacity_prompt
        self.compression_ratio = config.compression_ratio
        self.kernel_size = config.kernel_size
        self.pooling = config.pooling
   

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "window_size": self.window_size,
            "compression_ratio": self.compression_ratio,
            "kernel_size": self.kernel_size,
            "pooling": self.pooling,
        }
        cluster_name = ",".join(["snapkv"] + [str(i) for i in cluster_config.values()])
        return cluster_name, SnapKVCluster, cluster_config

       
class CAKECache(BaseKVCache):
    def __init__(self, config):
        super().__init__()
        self.window_size = config.window_size
        # self.max_capacity_prompt = config.max_capacity_prompt
        self.compression_ratio = config.compression_ratio
        self.kernel_size = config.kernel_size
        self.pooling = config.pooling
        self.pref_scores = []
        self.evict_scores = []
        self.layer_budget = []
        self.num_layers = config.num_layers
    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        # if prefill, then compress; if decode, then update
        # [bsz, num_heads, q_len, head_dim]

        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        query_states = cache_kwargs["query_states"]
        attention_mask = cache_kwargs["attention_mask"]
        num_key_value_groups = cache_kwargs["num_key_value_groups"]


        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        q_len = query_states.shape[-2]
        initializing_kv_cluster = False
        if (
            len(self.key_cache) == layer_idx
        ):  # initialize kv_cluster, ie, the first query/context
            initializing_kv_cluster = True
            self.compressed_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
                layer_idx,
            )
        else:  # the follow up queries/contexts
            
            if update_global_past_kv:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            else:  # add KVs to temp_kv_cache
                if len(self.temp_key_cache) == layer_idx:
                    self.temp_key_cache.append(key_states)
                    self.temp_value_cache.append(value_states)
                else:
                    self.temp_key_cache[layer_idx] = torch.cat(
                        [self.temp_key_cache[layer_idx], key_states], dim=-2
                    )
                    self.temp_value_cache[layer_idx] = torch.cat(
                        [self.temp_value_cache[layer_idx], value_states], dim=-2
                    )

        # torch.cuda.empty_cache()
        
        if not initializing_kv_cluster:  # return the compressed KV cache
            if self.temp_key_cache:  # concat global past_kv and temp_kv_cache
                key_states = torch.cat(
                    [self.key_cache[layer_idx], self.temp_key_cache[layer_idx]], dim=-2
                )
                value_states = torch.cat(
                    [self.value_cache[layer_idx], self.temp_value_cache[layer_idx]],
                    dim=-2,
                )
            else:
                key_states = self.key_cache[layer_idx]
                value_states = self.value_cache[layer_idx]
        
        return key_states, value_states
    def compressed_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        layer_idx: int,
    ):
        if self.kv_cluster_granularity == "layer":
            kv_cluster = self.get_kv_cluster_class(layer_idx)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            kv_cluster.update_kv(
                self,
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
            )

        else:
            assert (
                False
            ), f"kv_cluster_granularity {self.kv_cluster_granularity} not supported"
    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "window_size": self.window_size,
            "compression_ratio": self.compression_ratio,
            "kernel_size": self.kernel_size,
            "pooling": self.pooling,
        }
        cluster_name = ",".join(["cake"] + [str(i) for i in cluster_config.values()])
        return cluster_name, CAKECluster, cluster_config

class PyramidKVCache(SnapKVCache):
    def __init__(self, config):
        super().__init__(config)
        self.num_layers = config.num_layers

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "num_hidden_layers": self.num_layers,
            "window_size": self.window_size,
            "compression_ratio": self.compression_ratio,
            # "max_capacity_prompt": self.max_capacity_prompt,
            "kernel_size": self.kernel_size,
            "pooling": self.pooling,
            "layer_idx": layer_idx,
        }
        cluster_name = ",".join(
            ["pyramidv"] + [str(i) for i in cluster_config.values()]
        )
        return cluster_name, PyramidKVCluster, cluster_config


class StreamingLLMKVCache(SnapKVCache):
    def __init__(self, config):
        # n_local = config.max_capacity_prompt - config.n_init
        # n_init = config.n_init
        # config.window_size = n_local
        # config.max_capacity_prompt = n_local + n_init
        self.window_size = config.window_size
        # self.max_capacity_prompt = config.max_capacity_prompt
        self.compression_ratio = config.compression_ratio
        super().__init__(config)

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "window_size": self.window_size,
            "compression_ratio": self.compression_ratio,
        }
        cluster_name = ",".join(
            ["streamingllm"] + [str(i) for i in cluster_config.values()]
        )
        return cluster_name, StreamingLLMKVCluster, cluster_config


class DynamicCacheWithRepeat(DynamicCache):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_key_cache = []
        self.temp_value_cache = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        if update_global_past_kv:  # add KVs to global past_kv
            assert len(self.temp_key_cache) == 0 and len(self.temp_value_cache) == 0, (
                "when you updating global past_kv, make sure the temp_kv_cache is empty. "
                "User past_key_values.clear_temp_kv_cache() to empty the temp_kv_cache"
            )
            
            # prefilling
            if len(self.key_cache) == layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:  # decoding
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
        else:  # add KVs to temp_kv_cache, this is used when you have a common context but different query, the KVs of the query will be added to temp_kv_cache, and will be cleaned in the next query
            
            if len(self.temp_key_cache) == layer_idx:
                self.temp_key_cache.append(key_states)
                self.temp_value_cache.append(value_states)
            else:  # decoding
                self.temp_key_cache[layer_idx] = torch.cat(
                    [self.temp_key_cache[layer_idx], key_states], dim=-2
                )
                self.temp_value_cache[layer_idx] = torch.cat(
                    [self.temp_value_cache[layer_idx], value_states], dim=-2
                )

        if self.temp_key_cache:  # concat global past_kv and temp_kv_cache
            key_states, value_states = torch.cat(
                [self.key_cache[layer_idx], self.temp_key_cache[layer_idx]], dim=-2
            ), torch.cat(
                [self.value_cache[layer_idx], self.temp_value_cache[layer_idx]], dim=-2
            )
        else:
            key_states, value_states = (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
            )

        # repeat kv if needed
        query_states = cache_kwargs.get("query_states", None)
        if query_states is not None:
            key_states = repeat_kv(
                key_states, query_states.size(1) // key_states.size(1)
            )
            value_states = repeat_kv(
                value_states, query_states.size(1) // value_states.size(1)
            )
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[-1].shape[
                -2
            ]  # seq_len of temp_kv_cache
        self.temp_key_cache = []
        self.temp_value_cache = []
class MultiTurnDynamicCache(DynamicCache):
    def __init__(
        self,
    ):
        super().__init__()
        self.tail_k = []
        self.tail_v = []
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # if len(self.key_cache) == layer_idx:
        #         self.key_cache.append(key_states)
        #         self.value_cache.append(value_states)
        # else:  # decoding
        #         self.key_cache[layer_idx] = torch.cat(
        #             [self.key_cache[layer_idx], key_states], dim=-2
        #         )
        #         self.value_cache[layer_idx] = torch.cat(
        #             [self.value_cache[layer_idx], value_states], dim=-2
        #         )
        # return self.key_cache[layer_idx],self.value_cache[layer_idx]
        # Update the cache
        if key_states.shape[-2] != 1:
            if len(self.key_cache) == layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
        else:  # decoding
                if len(self.tail_k) == layer_idx:
                    self.tail_k.append(key_states)
                    self.tail_v.append(value_states)
                else:
                    self.tail_k[layer_idx] = torch.cat(
                    [ self.tail_k[layer_idx],key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [ self.tail_v[layer_idx],value_states], dim=-2
                    )
        if self.tail_k:
            key_states = torch.cat(
                        [self.key_cache[layer_idx], self.tail_k[layer_idx]], dim=-2
                    )
            value_states = torch.cat(
                        [self.value_cache[layer_idx], self.tail_v[layer_idx]],dim=-2,
                    )
        else:
            key_states, value_states = (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
            )
        return key_states,value_states
        
 
    def clear_temp_kv_cache(self):
        if self.tail_k:
            self._seen_tokens -= self.tail_k[-1].shape[-2]  # seq_len of tail
        self.tail_k = []
        self.tail_v = []
        
class HeteroCache(DynamicCache):
    def __init__(self, cache_config):
        super().__init__()
        self.config = cache_config
        self.num_attn_heads = cache_config.num_attn_heads
        self.num_kv_heads = cache_config.num_kv_heads
        self.num_layers = cache_config.num_layers
        self.gqa = self.num_kv_heads != self.num_attn_heads
        self.num_key_value_groups = self.num_attn_heads // self.num_kv_heads
        self.total_kv_heads = self.num_kv_heads * self.num_layers

        # Paged Attention configuration
        self.page_size = 256
        self.compression_ratio = cache_config.compression_ratio
        
        # Physical storage (indexed by layer)
        self.key_pools = []    
        self.value_pools = []  
        self.block_tables = [] 
        self.free_blocks = []  # List[torch.Tensor]
        self.free_block_pointers = [] # List[int]
        
        # Track how many contiguous blocks each layer's prefill phase occupies
        self.prefill_boundary_blocks = [] 
        self.layer_head_capacities = []
        # Logical metadata
        self.heads_len = []    
        self.head_indices = torch.arange(self.num_kv_heads, device="cuda")

        # Data and statistics
        self.data = cache_config.data
        self.full_heads = self.data['meta']['counts']['volatile'] + self.data['meta']['counts']['pivot']
        
        # CPU Offload related
        self.key_cache_cpu: List[torch.Tensor] = []
        self.value_cache_cpu: List[torch.Tensor] = []
        self.pinned_key_buffer = None
        self.pinned_value_buffer = None
        
        self.prefill_topk_set = []
        self.coefficient = [
            [[] for _ in range(len(count))]  
            for count in self.data['need_to_select_counts']
        ] 
        self.count = [
            [0 for _ in range(len(count))]  
            for count in self.data['need_to_select_counts']
        ] 
        
        self.query_states_buffer = [[] for _ in range(self.num_layers)]
        
        # Runtime parameters
        self.bsz = None
        self.head_dim = None
        self.prefill_len = None
        self._seen_tokens = 0
        
        self.decode_step = cache_config.decode_step
        self.stable_threshold = cache_config.stable_threshold
        self.device = "cpu" if cache_config.real_offload else "cuda"
        
        self.pooling = cache_config.pooling
        self.kernel_size = cache_config.kernel_size

        # Initialize placeholders
        for _ in range(self.num_layers):
            self.key_pools.append(None)
            self.value_pools.append(None)
            self.block_tables.append(None)
            self.free_blocks.append(None)
            self.free_block_pointers.append(0)
            self.heads_len.append(None)
            self.prefill_boundary_blocks.append(0)
            self.layer_head_capacities.append(0)

    def __bool__(self):
        return True

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if self._seen_tokens is None:
            return 0
        return self._seen_tokens

    def _init_layer_storage(self, layer_idx, device, dtype):
        """
        Compute the exact total number of blocks needed based on each head's full lifecycle (Prefill + Gen).
        """
        max_prefill_len = self.prefill_len
        max_gen_len = self.config.max_gen_len
        allocation_vec = self.data["allocation_matrix"][layer_idx]
        head_prefill_caps = []
        total_physical_blocks = 0
        total_prefill_blocks_allocated = 0
        
        for h in range(self.num_kv_heads):
            # Compute prefill capacity
            if h in self.data["need_to_keep_counts"][layer_idx]:
                cap = max_prefill_len
            else:
                ratio = allocation_vec[h]
                cap = int(self.base_compress_len * ratio)
                cap = min(cap, max_prefill_len)
                cap = max(cap, 0)
            
            prefill_blocks_needed = (cap + self.page_size - 1) // self.page_size
            aligned_cap = prefill_blocks_needed * self.page_size
            head_prefill_caps.append(aligned_cap)
            total_prefill_blocks_allocated += prefill_blocks_needed
            
            # Compute total blocks needed for the full lifecycle
            head_lifecycle_len = cap + max_gen_len
            head_total_blocks = (head_lifecycle_len + self.page_size - 1) // self.page_size
            total_physical_blocks += head_total_blocks
            
        # Allocate physical pool
        k_pool = torch.zeros((total_physical_blocks, self.page_size, 1, self.head_dim), 
                             device=device, dtype=dtype)
        v_pool = torch.zeros((total_physical_blocks, self.page_size, 1, self.head_dim), 
                             device=device, dtype=dtype)
        
        # Initialize block table
        max_logical_blocks = (max_prefill_len + max_gen_len + self.page_size - 1) // self.page_size + 1
        block_table = torch.full(
            (self.bsz * self.num_kv_heads, max_logical_blocks), -1,
            device=device, dtype=torch.int32
        )
        
        # Populate block table (prefill mapping)
        current_phys_block_ptr = 0
        for i in range(self.bsz * self.num_kv_heads):
            h_id = i % self.num_kv_heads
            allocated_tokens = head_prefill_caps[h_id]
            blocks_assigned = allocated_tokens // self.page_size
            
            if blocks_assigned > 0:
                indices = torch.arange(
                    current_phys_block_ptr, 
                    current_phys_block_ptr + blocks_assigned,
                    device=device, dtype=torch.int32
                )
                block_table[i, :blocks_assigned] = indices
                current_phys_block_ptr += blocks_assigned
        
        # Initialize free list (reserved for incremental decode-phase allocation)
        if current_phys_block_ptr > total_physical_blocks:
             raise RuntimeError(f"Layer {layer_idx}: Allocation logic error! Used more than allocated.")

        all_block_ids = torch.arange(total_physical_blocks, dtype=torch.int32, device=device)
        self.free_blocks[layer_idx] = all_block_ids[current_phys_block_ptr:].clone()
        
        self.free_block_pointers[layer_idx] = 0
        self.key_pools[layer_idx] = k_pool
        self.value_pools[layer_idx] = v_pool
        self.block_tables[layer_idx] = block_table
        self.heads_len[layer_idx] = torch.zeros(self.bsz * self.num_kv_heads, device=device, dtype=torch.int32)
        self.prefill_boundary_blocks[layer_idx] = total_prefill_blocks_allocated
        self.layer_head_capacities[layer_idx] = head_prefill_caps

    def _get_physical_content(self, layer_idx, head_idx, logical_len):
        """
        Gather logically contiguous KV data from paged memory.
        """
        seq_ids = torch.arange(self.bsz, device=self.device) * self.num_kv_heads + head_idx
        num_blocks = (logical_len + self.page_size - 1) // self.page_size
        block_ids = self.block_tables[layer_idx][seq_ids, :num_blocks].long()
        
        k_pool_flat = self.key_pools[layer_idx].view(-1, self.page_size, self.head_dim)
        gathered_flat = k_pool_flat[block_ids.flatten()]
        
        reconstructed = gathered_flat.view(self.bsz, num_blocks * self.page_size, self.head_dim)
        return reconstructed[:, :logical_len, :].unsqueeze(1)

    def prefill_select(self, query_states, key_states, value_states, layer_idx):
        """
        Revised: uses Masked Sort strategy to ensure compressed tokens maintain temporal order.
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_heads_to_compress = self.data['need_to_compress_counts'][layer_idx]
        
        if num_heads_to_compress == 0:
            return

        # 1. Prepare configuration
        ratios = torch.tensor(
            self.data["allocation_matrix"][layer_idx][:num_heads_to_compress], 
            device=query_states.device, dtype=torch.float32
        )
        caps = (self.base_compress_len * ratios).int()
        current_seq_len = key_states.shape[-2]
        keep_lens = torch.clamp(caps, max=current_seq_len)
        
        max_k = keep_lens.max().item()
        if max_k == 0:
            self.heads_len[layer_idx][:num_heads_to_compress] = 0
            return

        # 2. Batch attention computation
        k_subset = key_states[:, :num_heads_to_compress, :, :]
        num_q = num_heads_to_compress * self.num_key_value_groups if self.gqa else num_heads_to_compress
        q_subset = query_states[:, :num_q, -1:, :]

        if self.gqa:
            q_view = q_subset.view(bsz, num_heads_to_compress, self.num_key_value_groups, 1, head_dim)
            k_view = k_subset.view(bsz, num_heads_to_compress, 1, k_subset.shape[-2], head_dim)
            attn_weights = torch.matmul(q_view, k_view.transpose(-1, -2)) / math.sqrt(head_dim)
            attn_weights = attn_weights.view(bsz, num_q, 1, current_seq_len)
        else:
            attn_weights = torch.matmul(q_subset, k_subset.transpose(2, 3)) / math.sqrt(head_dim)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights.squeeze(2)

        if self.pooling == 'avgpool':
            attn_weights = F.avg_pool1d(attn_weights, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_weights = F.max_pool1d(attn_weights, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)

        if self.gqa:
            attn_weights = attn_weights.view(bsz, num_heads_to_compress, self.num_key_value_groups, -1)
            attn_weights = attn_weights.max(dim=-2).values
        
        # 3. Masked Sort strategy
        safe_k = min(max_k, attn_weights.shape[-1])
        topk_vals, topk_indices = attn_weights.topk(safe_k, dim=-1, sorted=True)
        
        range_grid = torch.arange(safe_k, device=query_states.device).view(1, 1, -1)
        keep_lens_view = keep_lens.view(1, -1, 1)
        
        excess_mask = range_grid >= keep_lens_view
        excess_mask = excess_mask.expand(bsz, -1, -1)

        # Sentinel padding: set excess position indices to infinity so they sort to the end
        INF_INDEX = current_seq_len + 1000
        topk_indices_masked = topk_indices.masked_fill(excess_mask, INF_INDEX)
        
        sorted_indices, _ = topk_indices_masked.sort(dim=-1)
        gather_indices_safe = sorted_indices.clamp(max=current_seq_len - 1)
        
        gather_idx = gather_indices_safe.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        selected_k = torch.gather(k_subset, 2, gather_idx)
        selected_v = torch.gather(value_states[:, :num_heads_to_compress, :, :], 2, gather_idx)

        # 4. Write to physical GPU memory
        valid_mask = ~excess_mask
        
        batch_ids = torch.arange(bsz, device=query_states.device).unsqueeze(1)
        head_ids = torch.arange(num_heads_to_compress, device=query_states.device).unsqueeze(0)
        seq_rows = batch_ids * self.num_kv_heads + head_ids 
        
        start_blocks = self.block_tables[layer_idx][seq_rows, 0].long()
        start_phys_offsets = start_blocks * self.page_size
        
        dest_indices_matrix = start_phys_offsets.unsqueeze(-1) + range_grid
        
        flat_selected_k = selected_k.view(-1, head_dim)
        flat_selected_v = selected_v.view(-1, head_dim)
        flat_dest_idx = dest_indices_matrix.view(-1)
        flat_mask = valid_mask.reshape(-1)
        
        final_src_k = flat_selected_k[flat_mask]
        final_src_v = flat_selected_v[flat_mask]
        final_dest_idx = flat_dest_idx[flat_mask]
        
        flat_k_pool = self.key_pools[layer_idx].view(-1, self.head_dim)
        flat_v_pool = self.value_pools[layer_idx].view(-1, self.head_dim)
        
        flat_k_pool.index_copy_(0, final_dest_idx, final_src_k)
        flat_v_pool.index_copy_(0, final_dest_idx, final_src_v)
        
        # 5. Update lengths
        lengths_update = keep_lens.unsqueeze(0).expand(bsz, -1).reshape(-1)
        flat_rows = seq_rows.view(-1)
        self.heads_len[layer_idx].index_put_((flat_rows,), lengths_update.int())

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if self.bsz is None:
            self.bsz = key_states.shape[0]
            self.head_dim = key_states.shape[-1]
        
        if not hasattr(self, "base_compress_len"):
            seq_len_in = key_states.shape[-2]
            self.prefill_len = seq_len_in
            self.base_compress_len = get_compress_len(
                self.compression_ratio, 
                self.full_heads, 
                seq_len_in, 
                self.total_kv_heads
            )
        if self.key_pools[layer_idx] is None:
            self._init_layer_storage(layer_idx, key_states.device, key_states.dtype)

        seq_len = key_states.shape[-2]

        # Prefill phase
        if seq_len > 1:
            self.prefill_len = seq_len
            heads_to_keep = self.data["need_to_keep_counts"][layer_idx]
            heads_to_update = self.data["need_to_update_counts"][layer_idx]
            
            flat_k_pool = self.key_pools[layer_idx].view(-1, self.head_dim)
            flat_v_pool = self.value_pools[layer_idx].view(-1, self.head_dim)
            
            if len(heads_to_keep) > 0:
                heads_to_keep_tensor = torch.tensor(heads_to_keep, device=key_states.device, dtype=torch.long)
                
                src_k = key_states.index_select(1, heads_to_keep_tensor)
                src_v = value_states.index_select(1, heads_to_keep_tensor)
                
                src_k_flat = src_k.reshape(-1, self.head_dim)
                src_v_flat = src_v.reshape(-1, self.head_dim)
                
                batch_ids = torch.arange(self.bsz, device=key_states.device).unsqueeze(1)
                head_ids_expanded = heads_to_keep_tensor.unsqueeze(0)
                seq_map_rows = (batch_ids * self.num_kv_heads + head_ids_expanded).view(-1)
                
                start_block_ids = self.block_tables[layer_idx][seq_map_rows, 0].long()
                start_token_offsets = start_block_ids * self.page_size
                
                range_offsets = torch.arange(seq_len, device=key_states.device).unsqueeze(0)
                dest_indices = start_token_offsets.unsqueeze(1) + range_offsets
                dest_indices_flat = dest_indices.view(-1)
                
                flat_k_pool.index_copy_(0, dest_indices_flat, src_k_flat)
                flat_v_pool.index_copy_(0, dest_indices_flat, src_v_flat)
                
                self.heads_len[layer_idx].index_fill_(0, seq_map_rows, seq_len)

            # CPU Offload Backup
            if self.device == "cpu":
                self.pinned_key_buffer = torch.empty(
                    (self.bsz, key_states.shape[-2], self.head_dim), 
                    dtype=key_states.dtype, device='cpu', pin_memory=True
                )
                self.pinned_value_buffer = torch.empty(
                    (self.bsz, key_states.shape[-2], self.head_dim), 
                    dtype=value_states.dtype, device='cpu', pin_memory=True
                )
                if len(heads_to_update) == 0:
                    self.key_cache_cpu.append([None])
                    self.value_cache_cpu.append([None])
                else:
                    gpu_key = key_states.detach()
                    cpu_key = torch.empty(gpu_key.shape, dtype=gpu_key.dtype, pin_memory=True)
                    cpu_key.copy_(gpu_key, non_blocking=True)
                    gpu_value = value_states.detach()
                    cpu_value = torch.empty(gpu_value.shape, dtype=gpu_value.dtype, pin_memory=True)
                    cpu_value.copy_(gpu_value, non_blocking=True)
                    self.key_cache_cpu.append(cpu_key)
                    self.value_cache_cpu.append(cpu_value)
            else:
                if len(heads_to_update) == 0:
                        self.key_cache_cpu.append(None)
                        self.value_cache_cpu.append(None)
                else:
                    self.key_cache_cpu.append(key_states.detach())
                    self.value_cache_cpu.append(value_states.detach())
            return key_states, value_states

        # Decode phase
        else:
            flat_k = key_states.view(-1, self.head_dim)
            flat_v = value_states.view(-1, self.head_dim)
            
            current_lens = self.heads_len[layer_idx] 
            logic_block_indices = current_lens // self.page_size
            page_offsets = current_lens % self.page_size
            
            total_seqs = flat_k.shape[0]
            seq_indices = torch.arange(total_seqs, device=key_states.device)
            current_phys_blocks = self.block_tables[layer_idx][seq_indices, logic_block_indices.long()]
            
            # Detect and allocate new blocks
            needs_alloc_mask = (current_phys_blocks == -1)
            num_new_blocks = needs_alloc_mask.sum().item()
            
            if num_new_blocks > 0:
                ptr = self.free_block_pointers[layer_idx]
                if ptr + num_new_blocks > self.free_blocks[layer_idx].size(0):
                    raise RuntimeError(f"Layer {layer_idx} OOM.")
                
                new_block_ids_gpu = self.free_blocks[layer_idx][ptr : ptr + num_new_blocks]
                self.free_block_pointers[layer_idx] += num_new_blocks
                
                alloc_seq_indices = seq_indices[needs_alloc_mask]
                alloc_col_indices = logic_block_indices[needs_alloc_mask].long()
                
                self.block_tables[layer_idx][alloc_seq_indices, alloc_col_indices] = new_block_ids_gpu
                current_phys_blocks[needs_alloc_mask] = new_block_ids_gpu

            self.key_pools[layer_idx][current_phys_blocks, page_offsets, 0, :] = flat_k
            self.value_pools[layer_idx][current_phys_blocks, page_offsets, 0, :] = flat_v
            
            self.heads_len[layer_idx] += 1
            return key_states, value_states

    def decode_select(self, query_states, layer_idx):
        """
        Cumulative batch mode: dynamic selection and update.
        """
        self.query_states_buffer[layer_idx].append(query_states)

        if len(self.query_states_buffer[layer_idx]) < self.decode_step:
            return

        batched_query = torch.cat(self.query_states_buffer[layer_idx], dim=2)
        self.query_states_buffer[layer_idx] = []

        leader_member_relations = self.data["leader_member_relations"][layer_idx]
        if len(leader_member_relations) == 0:
            return

        bsz, num_heads, q_len_batch, head_dim = batched_query.shape
        flat_pool_k = self.key_pools[layer_idx].view(-1, self.head_dim)
        flat_pool_v = self.value_pools[layer_idx].view(-1, self.head_dim)

        for idx, (leader, member) in enumerate(leader_member_relations.items()):
            if not member: continue
            head = int(leader)
            key_states = self._get_physical_content(layer_idx, head, self.prefill_len)

            # Batch attention computation
            if self.gqa:
                q_start = head * self.num_key_value_groups 
                q_subset = batched_query[:, q_start:q_start + self.num_key_value_groups, :, :]
                k_subset_T = key_states.view(bsz, 1, key_states.shape[-2], head_dim).transpose(-1, -2)
                attn_weights = torch.matmul(q_subset, k_subset_T) / math.sqrt(self.head_dim)
            else:
                q_slice = batched_query[:, head, :, :]
                k_subset_T = key_states.transpose(2, 3).squeeze(1)
                attn_weights = torch.matmul(q_slice, k_subset_T) / math.sqrt(self.head_dim)
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            if self.pooling == 'avgpool':
                original_shape = attn_weights.shape
                attn_weights = attn_weights.view(-1, original_shape[-1])
                attn_weights = F.avg_pool1d(attn_weights.unsqueeze(1), kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1).squeeze(1)
                attn_weights = attn_weights.view(original_shape)
            elif self.pooling == 'maxpool':
                original_shape = attn_weights.shape
                attn_weights = attn_weights.view(-1, original_shape[-1])
                attn_weights = F.max_pool1d(attn_weights.unsqueeze(1), kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1).squeeze(1)
                attn_weights = attn_weights.view(original_shape)

            if self.gqa:
                attn_weights = attn_weights.max(dim=1).values 

            # Batch overlap coefficient computation
            K = self.base_compress_len
            cur_indices = attn_weights.topk(K, dim=-1).indices
            
            ref_indices = self.prefill_topk_set[layer_idx][:, idx, :]
            ref_indices_expanded = ref_indices.unsqueeze(1)
            ref_sorted, _ = ref_indices_expanded.sort(dim=-1)
            
            ref_sorted_expand = ref_sorted.expand(-1, q_len_batch, -1).contiguous()
            idx_in_ref = torch.searchsorted(ref_sorted_expand, cur_indices)
            idx_in_ref = idx_in_ref.clamp(max=K - 1)
            
            found_vals = torch.gather(ref_sorted_expand, -1, idx_in_ref)
            matches = (found_vals == cur_indices)
            
            overlap_counts = matches.sum(dim=-1).float()
            step_coefficients = (overlap_counts / K).mean(dim=0)
            
            step_coeffs_list = step_coefficients.tolist()
            median_coefficient = statistics.median(step_coeffs_list)
            
            if median_coefficient < self.stable_threshold:
                # Trigger update
                self.prefill_topk_set[layer_idx][:, idx, :] = cur_indices[:, -1, :]
                
                lengths = [
                    min(int(self.base_compress_len * self.data["allocation_matrix"][layer_idx][i]), self.prefill_len) 
                    for i in member
                ]
                max_k_alloc = max(lengths)
                
                final_step_attn = attn_weights[:, -1, :]
                batch_indices = final_step_attn.topk(max_k_alloc, dim=-1).indices
                batch_indices, _ = batch_indices.sort(dim=-1)
                batch_indices = batch_indices.squeeze()

                if self.device == "cpu":
                    # CPU offload mode
                    batch_indices_cpu = batch_indices.cpu()
                    for loop_idx, (head_idx, num_tokens) in enumerate(zip(member, lengths)):
                        if num_tokens == 0: continue 
                        
                        _indices = batch_indices_cpu[:num_tokens]
                        k_out = self.pinned_key_buffer[:, :num_tokens, :]
                        v_out = self.pinned_value_buffer[:, :num_tokens, :]
                        
                        k_src_view = self.key_cache_cpu[layer_idx].select(1, head_idx)
                        v_src_view = self.value_cache_cpu[layer_idx].select(1, head_idx)
                        
                        torch.index_select(k_src_view, 1, _indices, out=k_out)
                        torch.index_select(v_src_view, 1, _indices, out=v_out)

                        # Async upload
                        for b_i in range(bsz):
                            seq_table_row = b_i * self.num_kv_heads + head_idx
                            start_phys_block = self.block_tables[layer_idx][seq_table_row, 0].item()
                            if start_phys_block == -1: continue

                            phys_ptr_start = start_phys_block * self.page_size
                            flat_pool_k[phys_ptr_start : phys_ptr_start + num_tokens].copy_(
                                k_out[b_i], non_blocking=True
                            )
                            flat_pool_v[phys_ptr_start : phys_ptr_start + num_tokens].copy_(
                                v_out[b_i], non_blocking=True
                            )
                else:
                    # GPU direct mode
                    for loop_idx, (head_idx, num_tokens) in enumerate(zip(member, lengths)):
                        if num_tokens == 0: continue
                        
                        _indices = batch_indices[:num_tokens]
                        k_src_view = self.key_cache_cpu[layer_idx].select(1, head_idx)
                        v_src_view = self.value_cache_cpu[layer_idx].select(1, head_idx)
                        
                        k_selected = torch.index_select(k_src_view, 1, _indices)
                        v_selected = torch.index_select(v_src_view, 1, _indices)
                        
                        for b_i in range(bsz):
                            seq_table_row = b_i * self.num_kv_heads + head_idx
                            start_phys_block = self.block_tables[layer_idx][seq_table_row, 0].item()
                            if start_phys_block == -1: continue
                            
                            phys_ptr_start = start_phys_block * self.page_size
                            flat_pool_k[phys_ptr_start : phys_ptr_start + num_tokens].copy_(
                                k_selected[b_i], non_blocking=True
                            )
                            flat_pool_v[phys_ptr_start : phys_ptr_start + num_tokens].copy_(
                                v_selected[b_i], non_blocking=True
                            )

    def init_coefficient(self, query_states, key_states, layer_idx):
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_heads_to_select = self.data['need_to_select_counts'][layer_idx]
        if len(num_heads_to_select) == 0:
            self.prefill_topk_set.append([]) 
            return 
        
        key_states = key_states[:, num_heads_to_select, :, :]

        if self.gqa:
            q_start = num_heads_to_select[0] * self.num_key_value_groups
            q_end = (num_heads_to_select[-1]+1) * self.num_key_value_groups
            num_q = q_end - q_start 
            query_states = query_states[:, q_start:q_end, -1:, :]
  
            query_states_view = query_states.view(bsz, len(num_heads_to_select), self.num_key_value_groups, 1, self.head_dim)
            key_states_view = key_states.view(bsz, len(num_heads_to_select), 1, key_states.shape[-2], self.head_dim)

            attn_weights = torch.matmul(query_states_view, key_states_view.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn_weights = attn_weights.view(bsz, num_q, 1, q_len)
        else:
            q_start = num_heads_to_select[0]
            q_end = num_heads_to_select[-1]
            query_states = query_states[:, q_start:q_end, -1:, :]
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights.squeeze(2)

        if self.pooling == 'avgpool':
            attn_weights = F.avg_pool1d(attn_weights, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_weights = F.max_pool1d(attn_weights, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
                        
        if self.gqa:
            attn_weights = attn_weights.view(attn_weights.shape[0], -1, self.num_key_value_groups, attn_weights.shape[-1])
            attn_weights = attn_weights.max(dim=-2).values
        k_val, k_idx = attn_weights.topk(self.base_compress_len, dim=-1, sorted=True)
        
        self.prefill_topk_set.append(k_idx)
            
    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[0].shape[-2]
        self.temp_key_cache = {}
        self.temp_value_cache = {}


