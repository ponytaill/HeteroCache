import os
import json
import random
import argparse
import time
import logging

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from heterocache.cache_utils import *
from heterocache.llama_model import LlamaForCausalLM_forward
from heterocache.utils import load_model_and_tokenizer
from tools.log import create_log, get_logger


def build_meaningful_prompt(tokenizer, target_len, needle_depth=0.2):
    """
    Quickly build a meaningful prompt of the specified length, containing a randomly inserted 'needle' (key information),
    with a question at the end to force the model to trigger long-range retrieval.
    """
    base_text = (
        "The history of natural language processing (NLP) generally started in the 1950s, "
        "although work can be found from earlier periods. In 1950, Alan Turing published an "
        "article titled 'Computing Machinery and Intelligence' which proposed what is now "
        "called the Turing test as a criterion of intelligence. The Georgetown experiment in 1954 "
        "involved fully automatic translation of more than sixty Russian sentences into English. "
        "The authors claimed that within three or five years, machine translation would be a solved problem. "
        "However, real progress was much slower, and after the ALPAC report in 1966, "
        "which found that ten-year-long research had failed to fulfill the expectations, "
        "funding for machine translation was dramatically reduced. "
    )

    secret_code = random.randint(10000, 99999)
    needle = f" The special pass key to the safe is {secret_code}. Remember this number carefully. "
    query = f" What is the special pass key to the safe? The special pass key is"
    

    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    

    current_len = len(needle_ids) + len(query_ids)
    remaining_len = target_len - current_len
    if remaining_len < 0:
        remaining_len = 0
        
    repeat_count = remaining_len // len(base_ids) + 1
    

    context_ids = base_ids * repeat_count

    context_ids = context_ids[:remaining_len]
    

    insert_idx = int(len(context_ids) * needle_depth)
    final_ids = context_ids[:insert_idx] + needle_ids + context_ids[insert_idx:] + query_ids

    if len(final_ids) > target_len:

        final_ids = final_ids[:target_len - len(query_ids)] + query_ids
    elif len(final_ids) < target_len:

        pad_len = target_len - len(final_ids)
        final_ids = final_ids + base_ids[:pad_len]

    return torch.tensor([final_ids])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--load_dir", type=str, default="./data/clusters")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default="None", help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str, default=None,choices=["HeteroCache","SnapKV","H2O","StreamingLLM","PyramidKV","FullKV","CAKE"])

    parser.add_argument("--nbits", type=int, default=4, help="")
    parser.add_argument("--compression_ratio", type=float, default=0.5, help="")
    parser.add_argument("--prefill_ratio", type=float, default=0.5, help="")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--quant", action="store_true",help="Enable 4bit quantization.")
    parser.add_argument("--quant_method", type=str, default=None,choices=["kivi"])
    parser.add_argument("--real_offload", action="store_true",help="Enable real offload.")
    parser.add_argument("--num_clusters", type=int, default=3, help="")
    parser.add_argument("--decode_step", type=int, default=5000, help="")
    parser.add_argument("--use_chat_format", action="store_true", help="")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format", help="")
    parser.add_argument("--topk", type=int, default=1024, help="")
    parser.add_argument("--stable_threshold", type=float, default=0.5)
    parser.add_argument("--sim_threshold", type=float, default=0.5)

    args = parser.parse_args()


    create_log(args)
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward
    args.max_gen_len = 100
    

    model, tokenizer = load_model_and_tokenizer(args)
    logger = get_logger()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # --- Warm Up ---
    logger.info(f"--- Warming up GPU...")
    # Simple text is sufficient for warmup, or build_meaningful_prompt can also be used
    warm_prompts = tokenizer(
        "Hi,who are you? I want you to tell me a funny story." * 20, 
        padding="longest", 
        return_tensors="pt", 
        add_special_tokens=True
    ).to(model.device)
    
    _ = model.generate(
        **warm_prompts,
        max_new_tokens=50,
    )
    torch.cuda.synchronize()
    print("Warm-up finished.")
    
 
    start_k = 32    # start length (K)
    end_k = 256      # end length (K)
    step_k = 32      # step size (K)
    KB_UNIT = 1024   # define the size of K
    # ===========================================


    test_lengths = [
        k * KB_UNIT
        for k in range(start_k, end_k + 1, step_k) # change +1 to +0 to exclude 256K
    ]

    for context_len in test_lengths:
        logger.info(f"---context_len {context_len}")
        
    
        input_ids = build_meaningful_prompt(tokenizer, target_len=context_len, needle_depth=0.2).to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)


        gen_cfg = GenerationConfig(
            temperature=0.1, 
            do_sample=True, 
            top_p=0.95, 
            max_new_tokens=50
        )
        

        try:
            torch.cuda.synchronize()
            start_time = time.time()
            
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=args.output_attentions,
                generation_config=gen_cfg,
                num_beams=1,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.eos_token_id,
            )
            
            torch.cuda.synchronize()
            end_time = time.time()

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM at context length {context_len}")
                torch.cuda.empty_cache()
            else:
                logger.error(f"Runtime error at context length {context_len}: {e}")

        del input_ids, attention_mask, output
        torch.cuda.empty_cache()

    logger.info("All tests finished.")