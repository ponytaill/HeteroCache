import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, AutoConfig
from transformers.cache_utils import DynamicCache 
from heterocache.cache_utils import *
import functools
from heterocache.llama_model import *
from heterocache.qwen_model import *

def reorder_qkv_weights_and_bias(weights: torch.Tensor, bias: torch.Tensor, num_heads: int, head_dim: int, r_head_indices: list) -> (torch.Tensor, torch.Tensor):
    """Reorder Q/K/V projection weights and bias by specified head indices."""
    output_dim, d_model = weights.shape
    if output_dim != num_heads * head_dim:
          raise ValueError(f"Weight output dimension mismatch. Expected {num_heads * head_dim}, got {output_dim}")

    all_head_indices = list(range(num_heads))
    nr_head_indices = [idx for idx in all_head_indices if idx not in r_head_indices]
    permutation_indices = r_head_indices + nr_head_indices

    reshaped_weights = weights.contiguous().view(num_heads, head_dim, d_model)
    reordered_weights = reshaped_weights[permutation_indices, :, :]
    final_weights = reordered_weights.contiguous().view(output_dim, d_model)

    final_bias = None
    if bias is not None:
        reshaped_bias = bias.contiguous().view(num_heads, head_dim)
        reordered_bias = reshaped_bias[permutation_indices, :]
        final_bias = reordered_bias.contiguous().view(output_dim)

    return final_weights, final_bias

def reorder_qkv_weights(weights: torch.Tensor, num_heads: int, head_dim: int, r_head_indices: list) -> torch.Tensor:
    """Reorder Q/K/V projection weights by specified head indices (weights only, no bias)."""
    output_dim, d_model = weights.shape
    if output_dim != num_heads * head_dim:
         raise ValueError(f"Weight output dimension mismatch. Expected {num_heads * head_dim}, got {output_dim}")

    all_head_indices = list(range(num_heads))
    nr_head_indices = [idx for idx in all_head_indices if idx not in r_head_indices]
    permutation_indices = r_head_indices + nr_head_indices

    reshaped_weights = weights.contiguous().view(num_heads, head_dim, d_model)
    reordered_weights = reshaped_weights[permutation_indices, :, :]
    final_weights = reordered_weights.contiguous().view(output_dim, d_model)

    return final_weights

def reorder_o_weights(weights: torch.Tensor, num_heads: int, head_dim: int, r_head_indices: list) -> torch.Tensor:
    """Reorder output projection (O) weights by specified head indices."""
    d_model, input_dim = weights.shape
    if input_dim != num_heads * head_dim:
        raise ValueError(f"Weight dimension mismatch. Expected input_dim {num_heads * head_dim}, got {input_dim}")

    all_head_indices = list(range(num_heads))
    nr_head_indices = [idx for idx in all_head_indices if idx not in r_head_indices]
    permutation_indices = r_head_indices + nr_head_indices

    reshaped_weights = weights.contiguous().view(d_model, num_heads, head_dim)
    reordered_weights = reshaped_weights[:, permutation_indices, :]
    final_weights = reordered_weights.contiguous().view(d_model, input_dim)

    return final_weights

def reorder_model_weights_by_classification(model, json_data: dict):
    """
    Reorder all attention weights in-place based on head classification from JSON.
    Priority Mapping:
      Type 1 (anchor)    -> Priority 0
      Type 4 (satellite) -> Priority 1
      Type 2 (volatile)  -> Priority 2
      Type 3 (pivot)     -> Priority 3
    Returns permutation records for each layer's KV heads.
    """
    d_model = model.config.hidden_size
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_layers = model.config.num_hidden_layers
    head_dim = d_model // num_q_heads
    is_gqa = num_q_heads != num_kv_heads

    if "classification_matrix" not in json_data:
        raise ValueError("JSON data must contain 'classification_matrix'.")

    classification_matrix = json_data["classification_matrix"]

    if len(classification_matrix) != num_layers:
        raise ValueError(f"Matrix layers ({len(classification_matrix)}) != Model layers ({num_layers})")

    print(f"Reordering model weights based on JSON classification...")
    # Modified comment to reflect new terminology
    print(f"Sort Order: Anchor (1) -> Satellite (4) -> Volatile (2) -> Pivot (3)")

    # 1: anchor, 4: satellite, 2: volatile, 3: pivot
    SORT_PRIORITY = {1: 0, 4: 1, 2: 2, 3: 3}
    permutation_records = []

    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc="Reordering Layers")):
        layer_types = classification_matrix[layer_idx]
        current_layer_types = layer_types[:num_kv_heads]

        if len(current_layer_types) != num_kv_heads:
             raise ValueError(f"Layer {layer_idx}: JSON contains {len(current_layer_types)} heads, but model has {num_kv_heads} KV heads.")

        indexed_types = []
        for old_idx, type_id in enumerate(current_layer_types):
            priority = SORT_PRIORITY.get(type_id, 99)
            indexed_types.append((old_idx, priority))

        indexed_types.sort(key=lambda x: x[1])
        sorted_kv_indices = [x[0] for x in indexed_types]
        permutation_records.append(sorted_kv_indices)

        sorted_q_indices = []
        if is_gqa:
            group_size = num_q_heads // num_kv_heads
            for kv_old_idx in sorted_kv_indices:
                start_q = kv_old_idx * group_size
                end_q = (kv_old_idx + 1) * group_size
                sorted_q_indices.extend(list(range(start_q, end_q)))
        else:
            sorted_q_indices = sorted_kv_indices

        model_dtype = model.dtype

        new_q_weight, new_q_bias = reorder_qkv_weights_and_bias(
            weights=layer.self_attn.q_proj.weight.to(model_dtype),
            bias=layer.self_attn.q_proj.bias.to(model_dtype) if layer.self_attn.q_proj.bias is not None else None,
            num_heads=num_q_heads,
            head_dim=head_dim,
            r_head_indices=sorted_q_indices
        )
        layer.self_attn.q_proj.weight.data.copy_(new_q_weight)
        if new_q_bias is not None:
            layer.self_attn.q_proj.bias.data.copy_(new_q_bias)

        new_k_weight, new_k_bias = reorder_qkv_weights_and_bias(
            weights=layer.self_attn.k_proj.weight.to(model_dtype),
            bias=layer.self_attn.k_proj.bias.to(model_dtype) if layer.self_attn.k_proj.bias is not None else None,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            r_head_indices=sorted_kv_indices
        )
        layer.self_attn.k_proj.weight.data.copy_(new_k_weight)
        if new_k_bias is not None:
            layer.self_attn.k_proj.bias.data.copy_(new_k_bias)

        new_v_weight, new_v_bias = reorder_qkv_weights_and_bias(
            weights=layer.self_attn.v_proj.weight.to(model_dtype),
            bias=layer.self_attn.v_proj.bias.to(model_dtype) if layer.self_attn.v_proj.bias is not None else None,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            r_head_indices=sorted_kv_indices
        )
        layer.self_attn.v_proj.weight.data.copy_(new_v_weight)
        if new_v_bias is not None:
            layer.self_attn.v_proj.bias.data.copy_(new_v_bias)

        layer.self_attn.o_proj.weight.data.copy_(
            reorder_o_weights(
                layer.self_attn.o_proj.weight.to(model_dtype),
                num_q_heads,
                head_dim,
                sorted_q_indices
            )
        )

    print("Model weights reordered successfully.")
    return permutation_records


def load_model_and_tokenizer(args):
    """
    Load model and tokenizer with optional KV cache compression, quantization, and weight reordering.
    Supports methods: FullKV, SnapKV, PyramidKV, StreamingLLM, H2O, HeteroCache (legacy alias: HeteroCache), CAKE, RocketKV, Brutal.
    """
    method = args.method.lower()
    method_to_cache_obj = {
        "fullkv": DynamicCache,
        "snapkv": SnapKVCache,
        "pyramidkv": PyramidKVCache,
        "streamingllm": StreamingLLMKVCache,
        "h2o": H2OKVCache,
        "heterocache": HeteroCache,
        "cake": CAKECache,
    }

    # Apply monkey patches for custom attention implementations
    if method in ["heterocache"]:
        transformers.models.llama.modeling_llama.LlamaAttention= LlamaAttention
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention= Qwen2Attention
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn_forward
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn_forward

    if "70b" in args.model_path.lower():
        transformers.models.llama.modeling_llama.LlamaMLP.forward = llama_mlp_forward
        transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = llama_rms_forward

    quant_config = None
    if args.quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    need_reorder = method in ["heterocache"]
    final_state_dict = None

    def get_reorder_data():
        model_name = args.model_path.lower().split("/")[-1]
        head_dir = os.path.join(args.load_dir, f"Head_Analysis_{model_name}_topk{args.topk}_th{args.stable_threshold}_sim{args.sim_threshold}_step{args.decode_step}.json")
        with open(head_dir, "r", encoding="utf-8") as f:
            return json.load(f)

    if need_reorder and args.quant:
        print("[Pre-process] Loading FP16 model for pre-quantization reordering...")
        model_temp = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=None
        )

        data = get_reorder_data()
        permutation_records = reorder_model_weights_by_classification(model_temp, data)

        final_state_dict = {k: v.cpu() for k, v in model_temp.state_dict().items()}

        del model_temp
        gc.collect()
        torch.cuda.empty_cache()
        print("[Pre-process] Reordered weights cached in memory. GPU cleared.")

    print(f"Loading final model (Quantization: {args.quant})...")

    if final_state_dict is not None:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

        try:
            arch_name = config.architectures[0]
            model_class = getattr(transformers, arch_name)
        except (AttributeError, IndexError):
            model_class = AutoModelForCausalLM

        print(f"Using specific model class: {model_class.__name__} to bypass AutoModel factory checks.")

        model = model_class.from_pretrained(
            None,
            config=config,
            state_dict=final_state_dict,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quant_config,
            attn_implementation=args.attn_implementation
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quant_config,
            attn_implementation=args.attn_implementation
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        padding_side="left"
    )


    if need_reorder and not args.quant:
        print("Applying in-memory reordering (No Quantization)...")
        data = get_reorder_data()
        permutation_records = reorder_model_weights_by_classification(model, data)

    cache_config = {}
    if method != "fullkv":
        common_params = {
            "pooling": "avgpool",
            "kernel_size": 5,
            "num_attn_heads": model.config.num_attention_heads,
            "num_kv_heads": model.config.num_key_value_heads,
            "num_layers": model.config.num_hidden_layers,
            "compression_ratio": args.compression_ratio
        }

        if method in ["heterocache"]:
            if "need_to_update_counts" not in data:
                data["need_to_update_counts"] = []
            if "need_to_select_counts" not in data:
                data["need_to_select_counts"] = []
            if "need_to_compress_counts" not in data:
                data["need_to_compress_counts"] = []
            if "need_to_keep_counts" not in data:
                data["need_to_keep_counts"] = []

            for layer_idx in range(model.config.num_hidden_layers):
                perm_indices = permutation_records[layer_idx]
                if hasattr(perm_indices, 'tolist'):
                    perm_indices = perm_indices.tolist()

                original_allocation = data["allocation_matrix"][layer_idx]
                original_classification_matrix = data["classification_matrix"][layer_idx]

                reordered_classification_matrix = np.array(original_classification_matrix)[permutation_records[layer_idx]].tolist()
                reordered_allocation = np.array(original_allocation)[permutation_records[layer_idx]].tolist()

                data["allocation_matrix"][layer_idx] = reordered_allocation
                data["classification_matrix"][layer_idx] = reordered_classification_matrix

                num_heads_to_select = reordered_classification_matrix.count(1) + reordered_classification_matrix.count(4)
                data["need_to_compress_counts"].append(num_heads_to_select)
                data["need_to_select_counts"].append([i for i, x in enumerate(reordered_classification_matrix) if x == 3])
                data["need_to_update_counts"].append([i for i, x in enumerate(reordered_classification_matrix) if x == 4])
                data["need_to_keep_counts"].append([i for i, x in enumerate(reordered_classification_matrix) if x == 2 or x == 3])

                original_relations = data["leader_member_relations"][layer_idx]
                old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(perm_indices)}
                new_layer_relations = {}

                for old_leader_str, old_members in original_relations.items():
                    old_leader = int(old_leader_str)

                    if old_leader in old_to_new_map:
                        new_leader = old_to_new_map[old_leader]
                        new_members = [old_to_new_map[m] for m in old_members if m in old_to_new_map]
                        new_members.sort()
                        new_layer_relations[str(new_leader)] = new_members

                data["leader_member_relations"][layer_idx] = new_layer_relations

            num_layers = model.config.num_hidden_layers
            num_kv_heads = model.config.num_key_value_heads * num_layers
            
            # --- MODIFICATION START: Updated keys for 'volatile' and 'pivot' ---
            full_heads = data['meta']['counts']['volatile'] + data['meta']['counts']['pivot']
            # --- MODIFICATION END ---
            
            compress_heads = model.config.num_hidden_layers * num_kv_heads - full_heads
            assert int(args.compression_ratio * num_kv_heads) > full_heads, f"Too many full KV cache heads retained (volatile+pivot) to satisfy the compression ratio; lower the ratio or adjust clustering hyperparameters"

            cache_config = HeteroCacheConfig(
                data=data,
                stable_threshold=args.stable_threshold,
                real_offload=args.real_offload,
                max_gen_len=args.max_gen_len,
                decode_step=args.steps,
                **common_params
            )
        elif method in ["snapkv", "pyramidkv", "streamingllm", "h2o", "cake"]:
            win_size = 32 if method == "cake" else 1
            cache_config = CompressionCacheConfig(
                window_size=win_size,
                **common_params
            )

        model.config.cache_config = cache_config

    def prepare_cache(method: str, config):
        cache_obj: Cache = method_to_cache_obj[method]
        def _prepare_cache_for_generation(
            self, generation_config, model_kwargs: Dict, *args, **kwargs
        ) -> bool:
            model_kwargs["past_key_values"] = cache_obj(config)
        return _prepare_cache_for_generation

    prepare_cache_func = prepare_cache(method, cache_config)
    model._prepare_cache_for_generation = prepare_cache_func.__get__(
        model, model.__class__
    )

    return model, tokenizer