import os, csv, json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
import gc
import torch
current_dir = os.getcwd()
sub_project_path = os.path.join(current_dir, 'heterocache')
sys.path.insert(0, sub_project_path)
from heterocache.utils import load_model_and_tokenizer
maxlen_map = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3":7950,
    "llama-3":7950,
    "llama3.1":128500,
    "llama-3.1":128500,
    "deepseek":90000,
    "qwen": 128000
}

template_0shot = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$
# You are very knowledgeable. An expert. Think and respond with confidence.  
# Format your response as follows: "The correct answer is (insert answer here, ONLY ABCD)"."""


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
      
    # if "llama3" or "llama-3.1" in model_name:
     
    #     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"  
    if any(k in model_name for k in ("llama","deepseek")):
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
        
    elif "mistral" in model_name:
        prompt = f'<s>[INST] {prompt} [/INST]'

    elif "qwen" in model_name:
        prompt = "".join(prompt)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
  
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt
# def extract_answer(response):
#     response = response.replace('*', '')
#     match = re.search(r'The correct answer is \(([A-D])\)', response)
#     if match:
#         return match.group(1)
#     else:
#         match = re.search(r'The correct answer is ([A-D])', response)
#         if match:
#             return match.group(1)
#         else:
#             return None
def extract_answer(response):
   
    response = response.replace('*', '').strip()
   
    patterns = [
        r'The correct answer is \(([A-D])\)',  
        r'The correct answer is ([A-D])',      
        
        r'\{([A-D])\}',                     
        
        r'\(([A-D])\)',                        

        r'\b([A-D])\b'
    ]


    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)  
            
    return None
def get_pred(data, args, fout):

   
    for item in tqdm(data):
        context = item['context']
       
        template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        # truncate
        model_path = args.model_path.lower()

        for key in maxlen_map:
            if key in model_path:
                model_max_len = maxlen_map[key]



        # input_ids = tokenizer.encode(prompt)
        input_ids = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to(model.device).input_ids
        if len(input_ids[0]) > model_max_len:
            # input_ids = input_ids[:model_max_len//2] + input_ids[-model_max_len//2:]
            # prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
            half = int(model_max_len/2)
            prompt = tokenizer.decode(input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(input_ids[0][-half:], skip_special_tokens=True)
        prompt = build_chat(tokenizer, prompt, args.model_path.lower())  
        # tokenized_prompts = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
        tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to(model.device)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        context_length = batch_input_ids.shape[-1]
        
        if "deepseek" in args.model_path.lower():
            output = model.generate(
                **tokenized_prompts,
                max_new_tokens=2048,
                num_beams=1,
                do_sample=True,
                temperature=0.5,
                top_p=0.95,
                repetition_penalty=1.1, 
                no_repeat_ngram_size=3,
                # do_sample=False,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            output = model.generate(
                **tokenized_prompts,
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
                # temperature=0.1,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.eos_token_id,
            )
     
        if output == '':
            continue
        output =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)[0]    
        
        if "deepseek" in args.model_path.lower():
            print(output)
            if "</think>" in output:
           
                output = output.split("</think>")[-1]
            else:
              
                output = "The correct answer is (A) "
        
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

        gc.collect()
        
        torch.cuda.empty_cache()
def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    
    method_label = "HeteroCache" if args.method.lower() in ["heterocache", "heterocache"] else args.method
    out_file = os.path.join(args.save_dir, args.model_path.split("/")[-1] + f"_{method_label}"+ ".jsonl")

    dataset = json.load(open('./data/longbenchv2/data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    
    get_pred(data, args, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="./results/longbenchv2")

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")

    parser.add_argument("--load_dir", type=str, default="./data/clusters")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default="../models/Llama-3.1-8B-Instruct", help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str, default=None,choices=["HeteroCache","SnapKV","H2O","StreamingLLM","PyramidKV","FullKV","CAKE","RocketKV","Brutal"])

    parser.add_argument("--nbits", type=int, default=4, help="")
    parser.add_argument("--compression_ratio", type=float, default=0.5, help="")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--quant", action="store_true",help="Enable 4bit quantization.")
    parser.add_argument("--real_offload", action="store_true",help="Enable real offload.")
    parser.add_argument("--decode_step", type=int, default=5000, help="")
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument("--topk", type=int, default=1024, help="")
    parser.add_argument("--stable_threshold", type=float, default=0.5)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
 
    args = parser.parse_args()
    if "deepseek" in args.model_path.lower():
        args.max_gen_len = 2048
    else:
        args.max_gen_len = 128
    model,tokenizer = load_model_and_tokenizer(args)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    main()