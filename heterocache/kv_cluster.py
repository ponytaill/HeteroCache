import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

from typing import List

import numpy as np
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache


# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
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



class PyramidKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, compression_ratio = 0.5, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None, merge = None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.compression_ratio = compression_ratio
        # assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if not hasattr(self, "max_capacity_prompt"):
            self.max_capacity_prompt = int(self.compression_ratio * key_states.shape[-2] / num_key_value_groups)
            # self.max_capacity_prompt = max(self.max_capacity_prompt,2048)
        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
            
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
    
       
        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
        
        # print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}")
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)


            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class SnapKVCluster():
    def __init__(self, window_size = 64, compression_ratio = 0.5, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        # self.max_capacity_prompt = max_capacity_prompt
        # assert self.max_capacity_prompt - self.window_size > 0
        self.compression_ratio = compression_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling


    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if not hasattr(self, "max_capacity_prompt"):
            self.max_capacity_prompt = int(self.compression_ratio * q_len / num_key_value_groups)
            self.max_capacity_prompt = max(self.max_capacity_prompt,2)
     
        if  self.max_capacity_prompt > q_len:
            return key_states, value_states
        else:
        
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)

            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
           
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            
            
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            return key_states, value_states

class CAKECluster():
    def __init__(self, window_size = 64, compression_ratio = 0.5, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        # self.max_capacity_prompt = max_capacity_prompt
        # assert self.max_capacity_prompt - self.window_size > 0
        self.compression_ratio = compression_ratio
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.tau1 = 1
        self.tau2 = 1
        self.gamma = 200.0
        
    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
    def calculate_entropy(self,attention_scores):
        attention_scores = attention_scores.to(torch.float32)
        entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
        entropy= entropy.to(dtype=torch.float32)
        return entropy
    def adjust_budgets(self,budget_list, total_budget, seq_len, layer_nums):

        budget_list = np.array(budget_list, dtype=int)
        # Limit the budget of all layers to not exceed seq_len
        excess = np.maximum(budget_list - seq_len, 0)
        budget_list = np.minimum(budget_list, seq_len)

        # Adjust excess budget
        total_excess = np.sum(excess)

        if total_excess > 0:

            valid_indices = budget_list < seq_len
            num_valid = np.sum(valid_indices)

            if num_valid > 0:
                
                distribute_per_layer = total_excess // num_valid
                remainder = total_excess % num_valid

                budget_list[valid_indices] += distribute_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] += 1

        # Ensure total budget equals total_budget
        current_total_budget = np.sum(budget_list)
        budget_diff = total_budget - current_total_budget

        if budget_diff != 0:
            if budget_diff > 0:
                valid_indices = budget_list < seq_len  
            else:
                valid_indices = budget_list > 1  

            num_valid = np.sum(valid_indices)

            if num_valid > 0:
                adjust_per_layer = abs(budget_diff) // num_valid
                remainder = abs(budget_diff) % num_valid

                if budget_diff > 0:
                    budget_list[valid_indices] += adjust_per_layer
                    budget_list[np.where(valid_indices)[0][:remainder]] += 1
                else:
                    budget_list[valid_indices] -= adjust_per_layer
                    budget_list[np.where(valid_indices)[0][:remainder]] -= 1

        return budget_list.tolist()

    def update_kv(self, past_key_values, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # print("q_len",q_len)
        if not hasattr(self, "max_capacity_prompt"):
            self.max_capacity_prompt = int(self.compression_ratio * q_len)
            self.max_capacity_prompt = max(self.max_capacity_prompt,2)

            self.total_size = (self.max_capacity_prompt-self.window_size) * past_key_values.num_layers
        # print("self.max_capacity_prompt",self.max_capacity_prompt)
        if  self.max_capacity_prompt > q_len:
            return key_states, value_states
        else:
            if query_states.shape[1] != key_states.shape[1]:
                key_states = repeat_kv(key_states, num_key_value_groups)
                value_states = repeat_kv(value_states, num_key_value_groups)
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            disp = self.calculate_entropy(attn_weights[:,:,-self.window_size:,:-self.window_size])
            
            var = torch.var(attn_weights[:,:,-self.window_size:,:-self.window_size],dim=-2).sum(0).sum(0).sum(0)
            if torch.isnan(disp) or torch.isnan(var):
                pref_score = 0.0
            else:
                # Values are valid, compute preference score as normal
                pref_score = (disp**(1/self.tau1)*var**(1/self.tau2)).cpu().numpy()
            # pref_score = (disp**(1/self.tau1)*var**(1/self.tau2)).cpu().numpy()
          
            #compute preference score and hh score
            attention_score = attn_weights[:, :, -self.window_size:, :] 

            attn_mean = attention_score.mean(dim = -2)
            attn_var = attention_score.var(dim = -2)
            attn_cache = attn_mean + self.gamma * attn_var
            attn_cache = attn_cache[:, :, :-self.window_size]
            attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=5//2, stride=1)

            attn_cache = attn_cache.reshape(bsz, int(attn_cache.shape[1] / num_key_value_groups), num_key_value_groups, -1)
            hh_score = attn_cache.max(dim=-2).values
            
            past_key_values.pref_scores.append(pref_score)
            past_key_values.evict_scores.append(hh_score)
            past_key_values.layer_budget.append(self.max_capacity_prompt-self.window_size)
            
            layer_budgets = [pref_score/sum(past_key_values.pref_scores)*self.total_size for pref_score in past_key_values.pref_scores]
            
            layer_budgets = self.adjust_budgets(layer_budgets, self.total_size, q_len-self.window_size, past_key_values.num_layers)
            
            
            layer_idx = 0
    
            for budget in layer_budgets:
                if budget>= q_len-self.window_size:
                    budget = q_len-self.window_size
                past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
                past_key_values.layer_budget[layer_idx]=budget
                layer_idx +=1
            # layer_idx = 0
            # if len(layer_budgets) == past_key_values.num_layers:
            #     for budget in layer_budgets:
            #         if budget>= q_len-self.window_size:
            #             budget = q_len-self.window_size
            #         past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
            #         past_key_values.layer_budget[layer_idx]=budget
            #         layer_idx +=1
            # print(layer_idx,past_key_values.layer_budget)
            # print(layer_idx,sum(past_key_values.layer_budget))
    def evcit_layer_kvcache(self, past_key_values, layer_idx, budget):

        bsz, num_key_value_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape

    
        hh_score = past_key_values.evict_scores[layer_idx]

        if budget> hh_score.shape[-1]:
            budget=hh_score.shape[-1]

        indices = hh_score.topk(budget, dim=-1).indices
        hh_score_compress = hh_score.gather(dim=2, index=indices)
        past_key_values.evict_scores[layer_idx] = hh_score_compress

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        k_cur = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
        v_cur = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        
        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states
        current_seq_len = key_states.shape[2]
        # print(f"Layer {layer_idx} cache size: {current_seq_len}")

        return past_key_values    

class H2OKVCluster():
    def __init__(self, window_size = 64, compression_ratio = 0.5, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.compression_ratio = compression_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # print(f"H2O max_capacity_prompt {self.max_capacity_prompt}")
        if not hasattr(self, "max_capacity_prompt"):
            self.max_capacity_prompt = int(self.compression_ratio * key_states.shape[-2] / num_key_value_groups)
            self.max_capacity_prompt = max(self.max_capacity_prompt,2)
            # self.max_capacity_prompt = max(self.max_capacity_prompt,2048)
        if self.max_capacity_prompt > q_len:
            return key_states, value_states
        else:
           
            total_attn_weights_sum = torch.zeros(
                bsz, num_heads,q_len, 
                device=query_states.device, 
                dtype=torch.float32 # use float32 for accumulation to ensure precision
            )

            # mask_shape = (q_len, q_len)
            # causal_mask_bool = torch.tril(torch.ones(mask_shape, dtype=torch.bool, device=query_states.device))
            # attn_mask = torch.zeros(mask_shape, dtype=query_states.dtype, device=query_states.device)
            # attn_mask.masked_fill_(causal_mask_bool == False, torch.finfo(query_states.dtype).min)
            # 3. Define chunk size
            chunk_size = max(1, int(q_len / 100))
            # chunk_size = 100

            # 4. Iterate over each query chunk
            for i, query_chunk in enumerate(torch.split(query_states, chunk_size, dim=2)):
                # Start and end positions of the current chunk in the full sequence
                start_idx = i * chunk_size
                end_idx = start_idx + query_chunk.shape[2]
                
                # Compute attention scores for the current chunk (shape: [B, H, chunk_size, q_len])
                attn_weights_chunk = torch.matmul(query_chunk, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                # attn_weights_chunk += attn_mask[None, None, start_idx:end_idx, :]
                attn_weights_chunk = torch.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32)

                chunk_sum = attn_weights_chunk.sum(dim=-2)  
          
                # Accumulate into the running sum
                total_attn_weights_sum += chunk_sum
       
            attn_cache = total_attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)



            k_past_compress = key_states.gather(dim = 2, index = indices)
            v_past_compress = value_states.gather(dim = 2, index = indices)

            return k_past_compress, v_past_compress

            # attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            # mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            # mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            # mask = mask.to(attn_weights.device)
            # attention_mask = mask[None, None, :, :]

            # attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            # # if self.pooling == 'avgpool':
            # #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # # elif self.pooling == 'maxpool':
            # #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # # else:
            # #     raise ValueError('Pooling method not supported')
            # attn_cache = attn_weights_sum
            # indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            # indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            # k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            # v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            # k_cur = key_states[:, :, -self.window_size:, :]
            # v_cur = value_states[:, :, -self.window_size:, :]
            # key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            # value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            # return key_states, value_states

class StreamingLLMKVCluster():
    def __init__(self, window_size = 64, compression_ratio = 0.5, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        # self.max_capacity_prompt = max_capacity_prompt
        # assert self.max_capacity_prompt - self.window_size > 0
        self.compression_ratio = compression_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")
        if not hasattr(self, "max_capacity_prompt"):
            self.max_capacity_prompt = int(self.compression_ratio * key_states.shape[-2])
            self.max_capacity_prompt = max(self.max_capacity_prompt,5)
            self.max_capacity_prompt = self.max_capacity_prompt - 4
            self.sink_token = 4
            # self.max_capacity_prompt = max(self.max_capacity_prompt,2048)
        if self.max_capacity_prompt > q_len:
            return key_states, value_states
        else:
                
            k_past_compress = key_states[:, :, :self.sink_token, :]
            v_past_compress = value_states[:, :, :self.sink_token, :]
            k_cur = key_states[:, :, -self.max_capacity_prompt:, :]
            v_cur = value_states[:, :, -self.max_capacity_prompt:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

