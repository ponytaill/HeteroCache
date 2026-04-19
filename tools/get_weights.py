from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import transformers
import torch, pathlib
import time
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
)
import math
import argparse
import numpy as np
import random
import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.get_logger(__name__)

def monkey_patch(args):
    """
    Apply monkey patch to the Transformers library.
    Replaces the attention forward functions of Llama and Qwen2 models
    to save attention weight matrices to disk during inference.
    """
    
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Save attention weights
        save_path = f"{args.tensor_save_dir}/attn_weights_{module.layer_idx}_{key_states.shape[-2]}.pt"
        torch.save(attn_weights.detach().cpu(), save_path)

        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    # Apply patch
    import transformers.models.llama.modeling_llama, transformers.models.qwen2.modeling_qwen2
    transformers.models.llama.modeling_llama.eager_attention_forward = eager_attention_forward
    transformers.models.qwen2.modeling_qwen2.eager_attention_forward = eager_attention_forward
    print("Monkey patch applied: eager_attention_forward now saves attention weights.")
    
    def qwen2_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` to using externally computed `position_embeddings`."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Save attention weights
        torch.save(attn_weights.detach().cpu(),f"{args.tensor_save_dir}/attn_weights_{self.layer_idx}_{key_states.shape[-2]}.pt")      

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def get_attn_weights(args):
    """
    Load the model and tokenizer, build the input prompt, and run model.generate.
    This triggers the save logic in the monkey patch, writing attention tensors to disk.
    """
    model_path = pathlib.Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        attn_implementation="eager", # must be eager to allow patching
    )
    model.eval()

    with open(args.dataset_path) as f:
        lines = [ln.strip() for ln in f][:args.num_lines]

    if "qwen" in args.model_path.lower():
        formatted_prompts = []
        for line in lines:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": "请你续写下面的文段" + line}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        batch = tokenizer(
            formatted_prompts,
            return_tensors='pt',
            max_length=args.max_length,
            truncation=True,
            padding=True,
        ).to(model.device)
    else:
        batch = tokenizer(
            lines,
            return_tensors='pt',
            max_length=args.max_length,
            truncation=True,
            padding=True,
        ).to(model.device)

    print(f"Running model.generate() to save attention tensors (Max new tokens: {args.max_new_tokens})...")
    with torch.no_grad():
        gen_out = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
            do_sample=False
        )

    generated_ids = gen_out.sequences
    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for i, t in enumerate(texts):
        print(f"[Sample {i}] {t}\n")
    print("Attention tensor saving complete.")

def get_head_topk(files: List[str], args, config, seq_len: int) -> (Optional[List[List[List[set]]]], int, int, int, bool):
    """
    Read tensor files for the specified sequence length, handle GQA (Grouped Query Attention) logic.
    Compute and return the set of top-K token indices attended to by each head.
    """
    if not files:
        return None, 0, 0, 0, False

    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)

    is_gqa = num_q_heads != num_kv_heads
    if is_gqa:
        assert num_q_heads % num_kv_heads == 0, "The number of Q heads must be divisible by the number of KV heads!"
        group_size = num_q_heads // num_kv_heads
        effective_heads = num_kv_heads
    else:
        effective_heads = num_q_heads

    batch_size = torch.load(files[0], map_location='cpu').size(0)
    L, H_eff = args.num_layers, effective_heads

    head_topk: List[List[List[set]]] = [
        [[set() for _ in range(batch_size)] for _ in range(H_eff)] for _ in range(L)
    ]

    for fp in files:
        layer_idx = int(pathlib.Path(fp).stem.split("_")[-2])
        
        try:
            expected_seq_len_from_filename = int(pathlib.Path(fp).stem.split('_')[-1])
        except ValueError:
            if "prefill" in pathlib.Path(fp).stem and seq_len == args.max_length:
                 expected_seq_len_from_filename = args.max_length
            else:
                print(f"Warning: cannot parse filename {fp}. Skipping.")
                continue 

        if expected_seq_len_from_filename != seq_len:
            continue

        attn = torch.load(fp, map_location='cpu') 
        B, _, _, file_seq_len = attn.shape
        
        if file_seq_len != seq_len:
             print(f"Warning: *internal tensor shape* {file_seq_len} of file {fp} does not match expected {seq_len}. Skipping this file.")
             continue

        if is_gqa:
            grouped_attn = attn.view(B, num_kv_heads, group_size, -1, seq_len)
            processed_attn = torch.max(grouped_attn, dim=2).values
        else:
            processed_attn = attn
        prefill_context_len = args.max_length 
        
        for h in range(H_eff):
            # Take the attention of the last query token over all key tokens
            vec = processed_attn[:, h, -1:, :prefill_context_len].sum(-2)
            if vec.shape[-1] == 0:
                continue

            # Use AvgPool to smooth the attention distribution
            vec = F.avg_pool1d(vec.float(), kernel_size=min(5, vec.shape[-1]), padding=min(5, vec.shape[-1])//2, stride=1)
            k = min(args.topk, vec.shape[-1])
            if k == 0:
                continue
            
            idx = torch.topk(vec, k, dim=-1).indices
            for b_idx in range(batch_size):
                head_topk[layer_idx][h][b_idx] = set(idx[b_idx].tolist())
    
    return head_topk, batch_size, L, H_eff, is_gqa

def plot_heatmap(data, title, xlabel, ylabel, save_path, annot=False, fmt=".1f", cmap="Blues", xticklabels=None, yticklabels=None, **kwargs):
    """
    General-purpose heatmap plotting function.
    Forces a square image, supports custom data range (vmin/vmax) and axis labels.
    """
    if data is None or data.size == 0:
        print(f"Warning: no data to plot: {title}")
        return

    # Compute side length uniformly to ensure a square canvas
    side_length = max(12, max(data.shape[0], data.shape[1]) * 0.5)
    
    plt.figure(figsize=(side_length, side_length))
    
    xtl = "auto" if xticklabels is None else xticklabels
    ytl = "auto" if yticklabels is None else yticklabels
    
    annot_kws = {"size": 6} if data.shape[0] > 20 or data.shape[1] > 20 else {"size": 8}
    
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, annot_kws=annot_kws, 
                linewidths=.5, linecolor='lightgray', xticklabels=xtl, yticklabels=ytl, 
                square=True, 
                **kwargs)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")

def visualize_intra_layer_per_layer_per_step(all_head_topk, loaded_decode_lens, L, H_eff, is_gqa, args):
    """
    Visualization task 1 (Intra-Layer):
    Compute the attention Top-K overlap coefficient between different heads within the same layer at the same time step.
    Generate one heatmap per layer for each decode step.
    """
    print("\n--- (1/2) Starting computation and visualization: intra-layer head overlap coefficients (Intra-Layer) ---")
    print(f"    Will generate {len(loaded_decode_lens) * L} charts for {len(loaded_decode_lens)} decode steps and {L} layers.")
    
    model_name = args.model_path.lower().split("/")[-1]
    vis_dir = f"./visualizations/{model_name}/1_Intra_Layer_Per_Layer_Step"
    pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)
    head_type = "KV-Group" if is_gqa else "Head"

    for seq_len in loaded_decode_lens:
        print(f"  Processing Decode Step: {seq_len}")
        head_topk_data, b, _, _, _ = all_head_topk[seq_len]
        
        for l_idx in range(L): 
            overlap = torch.zeros(H_eff, H_eff)
            for h1 in range(H_eff):
                for h2 in range(h1, H_eff): 
                    score = 0.0
                    for b_idx in range(b):
                        set1 = head_topk_data[l_idx][h1][b_idx]
                        set2 = head_topk_data[l_idx][h2][b_idx]
                        
                        if not set1 or not set2:
                            continue 
                        
                        inter = set1 & set2
                        denominator = min(len(set1), len(set2))
                        
                        if denominator > 0:
                            score += len(inter) / denominator
                    
                    if b > 0:
                        avg_score = score / b
                    else:
                        avg_score = 0.0
                        
                    overlap[h1, h2] = avg_score
                    overlap[h2, h1] = avg_score 
            
            title = f"Intra-Layer Overlap - Step {seq_len}, Layer {l_idx}\nModel: {model_name}, TopK: {args.topk}"
            save_path = f"{vis_dir}/intra_layer_step_{seq_len}_L{l_idx}.png"
            
            plot_heatmap(
                overlap.numpy(), 
                title, 
                f"{head_type} Index", 
                f"{head_type} Index", 
                save_path, 
                annot=True, 
                fmt=".2f", 
                vmin=0, 
                vmax=1
            )

    print(f"Intra-Layer visualization complete. Generated {len(loaded_decode_lens) * L} figures.")

def visualize_inter_step_per_head(all_head_topk, loaded_decode_lens, L, H_eff, is_gqa, args):
    """
    Visualization task 2 (Inter-Step):
    Compute the self-overlap coefficient of the same head across different time steps.
    Generate one heatmap per head per layer comparing all time step pairs.
    """
    print("\n--- (2/2) Starting computation and visualization: self-overlap coefficients of the same head across decode steps (Inter-Step) ---")
    print(f"    Will generate {L * H_eff} charts for {L} layers and {H_eff} heads.")
    
    model_name = args.model_path.lower().split("/")[-1]
    vis_dir = f"./visualizations/{model_name}/2_Inter_Step_Per_Head"
    pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)
    head_type = "KV-Group" if is_gqa else "Head"

    n_steps = len(loaded_decode_lens)
    step_list_str = [str(s) for s in loaded_decode_lens]

    for l_idx in range(L):
        print(f"  Processing Layer: {l_idx}/{L-1}")
        for h_idx in range(H_eff):
            
            overlap_matrix = torch.zeros(n_steps, n_steps)
            
            # Pre-extract data for this head across all steps
            head_data_all_steps = []
            batch_sizes = []
            for seq_len in loaded_decode_lens:
                head_topk_data, b, _, _, _ = all_head_topk[seq_len]
                head_data_all_steps.append(head_topk_data[l_idx][h_idx]) 
                batch_sizes.append(b)

            for i in range(n_steps):
                for j in range(i, n_steps):
                    
                    sets_i = head_data_all_steps[i] 
                    sets_j = head_data_all_steps[j] 
                    
                    b_i = batch_sizes[i]
                    b_j = batch_sizes[j]
                    current_batch_size = min(b_i, b_j)

                    score = 0.0
                    for b_idx in range(current_batch_size):
                        set1 = sets_i[b_idx]
                        set2 = sets_j[b_idx]
                        
                        if not set1 or not set2:
                            continue
                            
                        inter = set1 & set2
                        denominator = min(len(set1), len(set2))
                        
                        if denominator > 0:
                            score += len(inter) / denominator
                    
                    if current_batch_size > 0:
                        avg_score = score / current_batch_size
                    else:
                        avg_score = 0.0
                        
                    overlap_matrix[i, j] = avg_score
                    overlap_matrix[j, i] = avg_score

            title = f"Inter-Step Self-Overlap - Layer {l_idx}, {head_type} {h_idx}\nModel: {model_name}, TopK: {args.topk}"
            save_path = f"{vis_dir}/inter_step_L{l_idx}_H{h_idx}.png"
            
            plot_heatmap(
                overlap_matrix.numpy(), 
                title, 
                "Decode Step", 
                "Decode Step", 
                save_path, 
                annot=True, 
                fmt=".2f", 
                vmin=0, 
                vmax=1, 
                xticklabels=step_list_str, 
                yticklabels=step_list_str
            )

    print(f"Inter-Step visualization complete. Generated {L * H_eff} figures.")

def analyze_overlaps(args):
    """
    Main analysis entry function.
    Loads attention tensors for all time steps, extracts Top-K, and calls visualization subroutines.
    """
    config = AutoConfig.from_pretrained(pathlib.Path(args.model_path))
    
    prefill_len = args.max_length
    first_decode_step = args.max_length + 1
    last_decode_step = args.max_length + args.max_new_tokens
    
    all_seq_lens_to_load = [prefill_len] + list(range(first_decode_step, last_decode_step + 1))
    decode_lens = list(range(first_decode_step, last_decode_step + 1)) 
    
    if not decode_lens:
        print(f"Warning: max_new_tokens ({args.max_new_tokens}) is 0, no decode steps to analyze.")
        return

    all_head_topk = {}
    
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
    is_gqa = num_q_heads != num_kv_heads
    H_eff = num_kv_heads if is_gqa else num_q_heads
    L = args.num_layers
    
    print(f"Model config: {L} layers, {H_eff} effective heads (GQA={is_gqa})")
    print(f"Loading data for {len(all_seq_lens_to_load)} steps (Prefill: {prefill_len}, Decodes: {first_decode_step}..{last_decode_step})...")
    
    for seq_len in all_seq_lens_to_load:
        seq_len_str = str(seq_len)
        files = sorted(
            glob.glob(f"{args.tensor_save_dir}/attn_weights_*_{seq_len_str}.pt"),
            key=lambda p: int(pathlib.Path(p).stem.split("_")[-2]),
        )
        
        if seq_len == prefill_len and not files:
            seq_len_str_prefill_old = "prefill"
            files = sorted(
                glob.glob(f"{args.tensor_save_dir}/attn_weights_*_{seq_len_str_prefill_old}.pt"),
                key=lambda p: int(pathlib.Path(p).stem.split("_")[-2]),
            )

        if not files:
            print(f"Warning: no tensor files found for step {seq_len}. Skipping this step.")
            continue

        head_topk, b, _, _, _ = get_head_topk(files, args, config, seq_len)

        if head_topk is not None:
            all_head_topk[seq_len] = (head_topk, b, L, H_eff, is_gqa)
        else:
             print(f"Warning: failed to load top-k data for step {seq_len}.")

    if not all_head_topk:
        print("Error: failed to load data for any step. Analysis aborted.")
        return

    loaded_decode_lens = [l for l in decode_lens if l in all_head_topk]
    if not loaded_decode_lens:
         print("Error: data was loaded, but no *any* decode step data was loaded. Cannot analyze.")
         return

    print(f"Successfully loaded data for {len(all_head_topk)} steps. Starting analysis of {len(loaded_decode_lens)} decode steps...")

    visualize_intra_layer_per_layer_per_step(all_head_topk, loaded_decode_lens, L, H_eff, is_gqa, args)
    
    # visualize_inter_step_per_head(all_head_topk, loaded_decode_lens, L, H_eff, is_gqa, args)

    print(f"\nAnalysis complete. Total charts generated: { (len(loaded_decode_lens) * L) + (L * H_eff) }.")
    print(f"All visualizations saved to ./visualizations/{args.model_path.lower().split('/')[-1]}/ directory.")

def tensors_already_dumped(save_dir: str, num_layers: int, max_length: int, max_new_tokens: int) -> bool:
    """
    Check whether all required attention tensor files (Prefill + all decode steps) already exist on disk.
    Returns True if files are complete, False otherwise.
    """
    all_seq_lens = [max_length] + list(range(max_length + 1, max_length + max_new_tokens))
    
    for layer in range(num_layers):
        for seq_len in all_seq_lens:
            numeric_file = f"{save_dir}/attn_weights_{layer}_{seq_len}.pt"
            
            if seq_len == max_length:
                prefill_str_file = f"{save_dir}/attn_weights_{layer}_prefill.pt"
                if not (os.path.exists(numeric_file) or os.path.exists(prefill_str_file)):
                    print(f"Missing detected: Prefill file (L{layer}): {numeric_file} (or {prefill_str_file})")
                    return False
            else:
                if not os.path.exists(numeric_file):
                    print(f"Missing detected: Decode file (L{layer}): {numeric_file}")
                    return False
    return True

def set_seed(seed):
    """
    Set the global random seed to ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--model_path", type=str, default="../models/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="./data/wiki_demo.txt")
    parser.add_argument("--tensor_save_dir", type=str, default="./tensor")
    parser.add_argument("--topk", type=int, default=1024, help="Top-k attention indices to analyze")
    parser.add_argument("--num_layers", type=int, default=32, help="Number of layers (will be overridden by config)")
    parser.add_argument("--num_lines", type=int, default=1, help="Number of lines from dataset to process")
    parser.add_argument("--max_length", type=int, default=5000, help="Max sequence length for prefill")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate (decode steps)")

    args = parser.parse_args()

    set_seed(args.seed)
    model_name = args.model_path.lower().split("/")[-1]
    
    # Set tensor save directory
    args.tensor_save_dir = args.tensor_save_dir + f"/{model_name}_tensors"
    pathlib.Path(args.tensor_save_dir).mkdir(parents=True, exist_ok=True)

    # Get the actual number of layers
    config = AutoConfig.from_pretrained(pathlib.Path(args.model_path))
    args.num_layers = config.num_hidden_layers
    print(f"Model {model_name} has {args.num_layers} layers.")

    if tensors_already_dumped(args.tensor_save_dir, args.num_layers, args.max_length, args.max_new_tokens):
        print(f"Complete attention weight tensors detected in {args.tensor_save_dir}, skipping generation.")
    else:
        print("Complete tensors not detected, starting monkey-patch and generation.")
        monkey_patch(args)
        get_attn_weights(args)

    # analyze_overlaps(args)