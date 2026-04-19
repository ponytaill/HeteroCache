import torch
import argparse
import os
import glob
import pathlib
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig
from collections import defaultdict
import json
from tqdm import tqdm

def get_head_topk_exact(file_path, args, config):
    """
    Load an attention weight file, handle GQA logic, and compute the top-k position indices
    attended to by each attention head. Returns a list of topk index sets per head.
    """
    try:
        attn = torch.load(file_path, map_location='cpu')
    except Exception as e:
        return None, 0, 0

    if len(attn.shape) == 3:
        attn = attn.unsqueeze(2)
        
    B, num_saved_heads, _, file_seq_len = attn.shape

    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
    is_gqa = num_q_heads != num_kv_heads
    

    if is_gqa and num_saved_heads == num_q_heads:
        group_size = num_q_heads // num_kv_heads
        grouped_attn = attn.view(B, num_kv_heads, group_size, -1, file_seq_len)
        processed_attn = torch.mean(grouped_attn, dim=2)
        effective_heads = num_kv_heads
    else:
        processed_attn = attn
        effective_heads = num_saved_heads

    prefill_context_len = args.max_length
    head_topk_sets = [[set() for _ in range(B)] for _ in range(effective_heads)]
    
    for h in range(effective_heads):
        vec = processed_attn[:, h, -1:, :prefill_context_len].sum(-2)
        if vec.shape[-1] == 0: continue
        
        k_size = min(5, vec.shape[-1])
        pad = k_size // 2
        vec = F.avg_pool1d(vec.float(), kernel_size=k_size, padding=pad, stride=1)
        
        k = min(args.topk, vec.shape[-1])
        if k == 0: continue
        
        idx = torch.topk(vec, k, dim=-1).indices
        for b_idx in range(B):
            head_topk_sets[h][b_idx] = set(idx[b_idx].tolist())
            
    return head_topk_sets, B, effective_heads

def calculate_overlap_ratio(set_a, set_b, topk):
    """
    Compute the overlap ratio of two sets (Intersection / Size).
    """
    if not set_a or not set_b: return 0.0
    inter = set_a.intersection(set_b)
    denominator = len(set_a) if len(set_a) > 0 else topk
    return len(inter) / denominator

def find_star_clusters_greedy(adjacency_dict, all_heads_list):
    """
    Build star clusters using a greedy strategy.
    Prioritizes nodes with the highest degree as Pivot (Leader), treating their neighbors as Satellites (Members).

    Returns:
        cluster_info: {pivot_idx: [satellite_idx, ...]}
        assigned_nodes: set of nodes successfully assigned as Pivot or Satellite
    """
    unassigned = set(all_heads_list)
    clusters_info = {} 
    assigned_nodes = set()

    while unassigned:
        candidates = []
        for n in unassigned:
          
            neighbors = adjacency_dict.get(n, set())
            valid_neighbors = neighbors.intersection(unassigned)
            degree = len(valid_neighbors)
            candidates.append((degree, n, valid_neighbors))
        
        if not candidates:
            break
            

        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        best_degree, leader, members_set = candidates[0]
        
 
        if best_degree == 0:
            break 
            
        members = sorted(list(members_set))
        clusters_info[leader] = members
        
        unassigned.remove(leader)
        assigned_nodes.add(leader)
        
        for m in members:
            unassigned.remove(m)
            assigned_nodes.add(m)
            
    return clusters_info, assigned_nodes

def analyze_combined_heads(args):
    """
    Main analysis function.
    1. Scan tensor files.
    2. Compute temporal stability (anchor/volatile) for each head.
    3. Build a spatial similarity graph and perform star clustering (pivot/satellite).
    4. Classify attention heads and save results.
    """
    print(f"Loading config from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    total_heads = config.num_attention_heads 
    
    classification_matrix = [[0] * total_heads for _ in range(num_layers)]
    allocation_matrix = [[-1.0] * total_heads for _ in range(num_layers)]
    
    leader_member_relations = [{} for _ in range(num_layers)]


    TYPE_MAP = {
        "anchor": 1,      
        "volatile": 2,    
        "pivot": 3,       
        "satellite": 4    
    }
    
    counts = {k: 0 for k in TYPE_MAP.keys()}

    print(f"Scanning tensor files in {args.tensor_save_dir}...")
    layer_files = defaultdict(dict)
    all_steps = set()
    pattern = os.path.join(args.tensor_save_dir, "attn_weights_*_*.pt")
    files = glob.glob(pattern)
    
    if not files:
        print("No tensor files found.")
        return

    for f in files:
        try:
            stem = pathlib.Path(f).stem
            parts = stem.split('_')
            layer_idx = int(parts[2])
            step = int(parts[3])
            layer_files[layer_idx][step] = f
            all_steps.add(step)
        except Exception: continue

    sorted_steps = sorted(list(all_steps))
    if not sorted_steps:
        print("Could not parse steps.")
        return

    prefill_step = sorted_steps[0]
    decode_steps = sorted_steps[1:]
    spatial_target_step = args.decode_step if args.decode_step in all_steps else sorted_steps[-1]
    
    print(f"Ref Step (Prefill): {prefill_step}")
    print(f"Spatial Check Step: {spatial_target_step}")
    print("-" * 50)

    for layer_idx in range(num_layers):
        if layer_idx not in layer_files or prefill_step not in layer_files[layer_idx]:
            continue

        print(f"Processing Layer {layer_idx:02d}...", end="\r")

        # A. Load Prefill data
        prefill_path = layer_files[layer_idx][prefill_step]
        base_sets, batch_size, num_heads = get_head_topk_exact(prefill_path, args, config)
        if base_sets is None: continue
        
        if num_heads != total_heads and layer_idx == 0:
             classification_matrix = [[0] * num_heads for _ in range(num_layers)]
             allocation_matrix = [[-1.0] * num_heads for _ in range(num_layers)]

        # B. Compute temporal dynamics
        head_dynamic_state = {} 
        head_median_scores = {} 
        
        head_overlaps_history = defaultdict(list)
        for step in decode_steps:
            if step not in layer_files[layer_idx]: continue
            step_path = layer_files[layer_idx][step]
            try:
                curr_sets, _, _ = get_head_topk_exact(step_path, args, config)
            except: continue
            if curr_sets is None: continue
            for h in range(num_heads):
                batch_overlaps = []
                for b in range(batch_size):
                    ov = calculate_overlap_ratio(base_sets[h][b], curr_sets[h][b], args.topk)
                    batch_overlaps.append(ov)
                avg_step_overlap = sum(batch_overlaps) / len(batch_overlaps) if batch_overlaps else 0
                head_overlaps_history[h].append(avg_step_overlap)
        
        dyn_threshold = args.stable_threshold 
        for h in range(num_heads):
            history = head_overlaps_history[h]
            if not history:
                dyn_type = "volatile" 
                median_score = 0.0
            else:
                median_score = np.median(history)
                dyn_type = "volatile" if median_score < dyn_threshold else "anchor"
            head_dynamic_state[h] = dyn_type
            head_median_scores[h] = float(median_score)

        # C. Build spatial adjacency graph
        spatial_path = layer_files[layer_idx][spatial_target_step]
        spatial_sets, _, _ = get_head_topk_exact(spatial_path, args, config)
        
        layer_adjacency = defaultdict(set)
        
        if spatial_sets is not None:
            for i in range(num_heads):
                for j in range(num_heads):
                    if i == j: continue
                    batch_sim_sum = 0.0
                    for b in range(batch_size):
                        sim = calculate_overlap_ratio(spatial_sets[i][b], spatial_sets[j][b], args.topk)
                        batch_sim_sum += sim
                    avg_sim = batch_sim_sum / batch_size
                    
                    if avg_sim >= args.sim_threshold:
                        layer_adjacency[i].add(j)

        # D. Run greedy star clustering
        all_heads = list(range(num_heads))
        clusters_info, assigned_nodes = find_star_clusters_greedy(layer_adjacency, all_heads)
        
        leader_member_relations[layer_idx] = clusters_info

        # E. Final type assignment
        for h in range(num_heads):
            allocation_matrix[layer_idx][h] = head_median_scores[h]

            # Case 1: it is a Pivot (Leader)
            if h in clusters_info:
                classification_matrix[layer_idx][h] = TYPE_MAP["pivot"]
                counts["pivot"] += 1
                allocation_matrix[layer_idx][h] = -1.0  # Pivot does not participate in ratio calculation

            # Case 2: it is a Satellite (Member)
            elif h in assigned_nodes:
                classification_matrix[layer_idx][h] = TYPE_MAP["satellite"]
                counts["satellite"] += 1

            # Case 3: it is Unique (anchor or volatile)
            else:
                original_dyn = head_dynamic_state[h]
                if original_dyn == "anchor":
                    classification_matrix[layer_idx][h] = TYPE_MAP["anchor"]
                    counts["anchor"] += 1
                else:
                    classification_matrix[layer_idx][h] = TYPE_MAP["volatile"]
                    counts["volatile"] += 1
                    allocation_matrix[layer_idx][h] = -1.0

    # F. Post-processing: compute Allocation Ratio
    print("\n" + "="*60)
    print("CALCULATING ALLOCATION RATIOS")
    print("="*60)
    
    target_scores = []
    for r in range(len(allocation_matrix)):
        for c in range(len(allocation_matrix[r])):
            val = allocation_matrix[r][c]
            if val > -0.5: 
                target_scores.append(val)
    
    num_targets = len(target_scores)
    total_inverse_score = 0.0
    epsilon = 1e-6
    
    for score in target_scores:
        total_inverse_score += (1.0 / (score + epsilon))
        
    print(f"Total Heads for Allocation: {num_targets}")
    
    if num_targets > 0 and total_inverse_score > 1e-9:
        for r in range(len(allocation_matrix)):
            for c in range(len(allocation_matrix[r])):
                score = allocation_matrix[r][c]
                if score > -0.5: 
                    inv_weight = 1.0 / (score + epsilon)
                    raw_allocation = inv_weight / total_inverse_score
                    ratio = raw_allocation * num_targets
                    allocation_matrix[r][c] = ratio
    else:
        print("Warning: No target heads found.")

    # Save final JSON
    model_name = os.path.basename(os.path.normpath(args.model_path)).lower()
    json_filename = f"./data/clusters/Head_Analysis_{model_name}_topk{args.topk}_th{args.stable_threshold}_sim{args.sim_threshold}_step{args.decode_step}.json"
    
    final_json_data = {
        "meta": {
            "type_map": TYPE_MAP,
            "counts": counts,
        },
        "classification_matrix": classification_matrix,
        "allocation_matrix": allocation_matrix,
        "leader_member_relations": leader_member_relations
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(final_json_data, f, indent=4)
        
    print("\n" + "="*60)
    print(f"Analysis Complete.")
    print(f"Type Counts: {counts}")
    print(f"Results saved to: '{json_filename}'")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_save_dir", type=str, default="./tensor")
    parser.add_argument("--model_path", type=str, default="../models/Llama-3.1-8B-Instruct")
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=5000)
    parser.add_argument("--stable_threshold", type=float, default=0.5)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--decode_step", type=int, default=5000)
    
    args = parser.parse_args()
    args.tensor_save_dir = os.path.join(args.tensor_save_dir, args.model_path.lower().split('/')[-1] + "_tensors")
    analyze_combined_heads(args)