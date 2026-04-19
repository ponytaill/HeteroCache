import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from heterocache.cache_utils import *
from heterocache.utils import load_model_and_tokenizer
datasets = ["narrativeqa", "qasper", "multifieldqa_en","hotpotqa" , "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en","lcc", "repobench-p"]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3":7950,
    "llama-3":7950,
    "llama3.1":128500,
    "llama-3.1":128500,
    "llama":128500,
    "deepseek":128500,
    "qwen": 128500
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
# Updated build_chat function
def build_chat(tokenizer, prompts, model_name):
    # Ensure prompts is a list; convert if a single string is passed
    if isinstance(prompts, str):
        prompts = [prompts]
        
    new_prompts = []
    
    for prompt_text in prompts:
        if any(k in model_name for k in ("llama3", "llama-3", "llama-3.1", "deepseek")):
      
            messages = [
                {"role": "user", "content": prompt_text},
            ]

            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            new_prompts.append(formatted)
            
        elif "llama2" in model_name:
            new_prompts.append(f"[INST]{prompt_text}[/INST]")
            
        elif "mistral" in model_name:
            new_prompts.append(f'<s>[INST] {prompt_text} [/INST]')
            
        elif "qwen" in model_name:
            # Qwen processing logic adapted for a single prompt
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            new_prompts.append(formatted)
        else:
            # No matching model found; keep prompt as-is
            new_prompts.append(prompt_text)

    return new_prompts

# def build_chat(prompt):
#         prompt = f"[INST] {prompt} [/INST]"
#         return prompt

# def build_prompt(prompt, dataset):
    
#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):
    

    print("Loading data...")
    
    test_data = []
    
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
    input_max_len = 0
    
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]


    
    output_max_len = dataset2maxlen[args.dataset]
   
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            
            length = example["length"]

            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            example["prompt"] = prompt
                
            test_data.append(example)

    print(f"Max Length is {input_max_len}")

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    
    for example in test_data:
        
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    
    model_name = model_path.split("/")[-1]
    
    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.compression_ratio}", args.dataset), exist_ok=True)
    method_label = "HeteroCache" if args.method.lower() == "heterocache" else args.method
    if args.method.lower() == "heterocache":
        fout = open(os.path.join(args.save_dir, f"{model_name}_{args.compression_ratio}", args.dataset, f"{method_label}_topk_{args.stable_threshold}_d_{args.sim_threshold}_m_{args.steps}_{args.real_offload}.json"), "w")
    else:
        fout = open(os.path.join(args.save_dir, f"{model_name}_{args.compression_ratio}", args.dataset, f"{method_label}.json"), "w")

   

    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        batch_prompts = prompts[i:i+args.eval_batch_size]
        
        # idx = batch_prompts[0].find("claim, premise, backing, rebuttal, and refutation")
        # print(batch_prompts[0][48388:48437])
       
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        
        batch_lengths = lengths[i:i+args.eval_batch_size]
        
        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]

        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to(model.device)

        batch_input_ids = tokenized_prompts.input_ids
        
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            batch_prompts = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
       
        if args.dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            batch_prompts = build_chat(tokenizer, batch_prompts, args.model_path.lower())  

        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to(model.device)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        

        context_length = batch_input_ids.shape[-1]
        
        # st = time.time()
        # if args.dataset in ["gov_report", "qmsum", "multi_news",]:
        #     output = model.generate(
        #         **tokenized_prompts,
        #         output_attentions = args.output_attentions,
        #         max_new_tokens=output_max_len,
        #         num_beams=1,
        #         do_sample=True,  # Enable sampling
        #         temperature=0.8, # Set desired temperature
        #         top_p=0.95,      # Set desired nucleus sampling (top-p) value
        #         min_length=context_length+1,
        #         eos_token_id=[tokenizer.eos_token_id],
        #         pad_token_id=tokenizer.eos_token_id,
        #     )
        # else:
        if "deepseek" in model_path :
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len + 10000,
                num_beams=1,
                do_sample=True,  # Enable sampling
                temperature=0.6, # Set desired temperature
                top_p=0.95,      # Set desired nucleus sampling (top-p) value
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.eos_token_id,
            )
            
        else:
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                # do_sample=True,  # Enable sampling
                # temperature=0.8, # Set desired temperature
                # top_p=0.95,      # Set desired nucleus sampling (top-p) value
                do_sample=False,
                # temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.eos_token_id,
            )
            
        
   
        batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
        # print(f"debbug batch_outputs {batch_outputs}")
        batch_generations = batch_outputs
        
        for j in range(args.eval_batch_size):

            if "deepseek" in model_path:
                if "</think>" in batch_generations[j]:
                    # Extract content after </think>
                    batch_generations[j] = batch_generations[j].split("</think>")[-1]
                else:
                    # No </think> found; mark as failed
                    batch_generations[j] = "This generation failed. "
            
            example = {}
            
            # example["prompt"] = batch_prompts[j]
            # example["input"] = batch_inputs[j]
            # example["context"] = batch_contexts[j]
            example["pred"] = batch_generations[j]
            example["answers"] = batch_answerss[j]
            example["length"] = batch_lengths[j]
            
            # example["dataset"] = batch_datasets[j]
            # example["language"] = batch_languages[j]
            example["all_classes"] = batch_all_classess[j]
            # example["_id"] = batch__ids[j]

            # print(f'{batch_generations[j]}')
            fout.write(json.dumps(example) + "\n")
        
    

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./results/longbench")
    parser.add_argument("--load_dir", type=str, default="./data/clusters")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
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

    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--quant", action="store_true",help="Enable 4bit quantization.")
    parser.add_argument("--real_offload", action="store_true",help="Enable real offload.")
    parser.add_argument("--decode_step", type=int, default=5000, help="")

    parser.add_argument("--topk", type=int, default=1024, help="")
    parser.add_argument("--stable_threshold", type=float, default=0.5)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    set_seed(args.seed)

    model = None 
    for idx, dataset in enumerate(datasets):
        
        if model is not None:
            del model     
            del tokenizer 
            gc.collect()  
            torch.cuda.empty_cache() 
        args.dataset = dataset
        args.data_file = f"data/longbench/{args.dataset}.jsonl"
        output_max_len = dataset2maxlen[args.dataset]
        if "deepseek" in args.model_path.lower():
            args.max_gen_len = output_max_len + 10000
        else:
            args.max_gen_len = output_max_len
        model,tokenizer = load_model_and_tokenizer(args)

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        model.eval()
        main(args)
