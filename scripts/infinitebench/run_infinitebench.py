import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))

from transformers import AutoTokenizer, GenerationConfig
from infinitebench.eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
import argparse
import torch
MAX_POSITION_ID = 128500 
TRUNCATE_LEN = 128500
GENERATION_CONFIG = None
DEFAULT_PATH = _HERE

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3":7950,
    "llama-3":7950,
    "llama3.1":128500,
    "llama-3.1":128500,
    "qwen": 128500
}

def setup_heterocache_path(path):
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        print(f"Adding HeteroCache path: {abs_path}")
        sys.path.insert(0, abs_path)

def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
      
    # if "llama3" or "llama-3.1" in model_name:
     
    #     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"  
    if any(k in model_name for k in ("llama3", "llama-3", "llama-3.1")):
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
def get_pred(
        args,
        model,
        tok: AutoTokenizer,
        input_text: str,
        max_tokens: int,
        verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    # output = model(input_text, generation_config=GENERATION_CONFIG,
    #                skip_special_tokens=True)

    input_text = build_chat(tokenizer, input_text, args.model_path.lower())  
    tokenized_prompts = tokenizer(input_text, padding="longest", return_tensors="pt", add_special_tokens=True).to(model.device)
    batch_input_ids = tokenized_prompts.input_ids
    attention_mask = tokenized_prompts.attention_mask
    

    context_length = batch_input_ids.shape[-1]
    output = model.generate(
            **tokenized_prompts,
            max_new_tokens=max_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id],
            pad_token_id=tokenizer.eos_token_id,
            min_length=context_length+1
        )
    output = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
    print("Chunked generation:", output)
    return output


if __name__ == "__main__":
    
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        # default="code_debug",
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default='./data/infinitebench/',
        help="The directory of data."
    )
    p.add_argument("--output_dir", type=str, default="./results/infinitebench", help="Where to dump the prediction results.")  # noqa
    p.add_argument(
        "--model_path",
        type=str,
        default="../models/Llama-3.1-8B-Instruct",
        help="The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it wll default to the official HF path.",  # noqa
    )  # noqa
    p.add_argument(
        "--model_name",
        type=str,
        default="Llama-3.1-8B-Instruct",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.")  # noqa
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.")  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    p.add_argument("--heterocache_path", type=str, default=DEFAULT_PATH, help="Path to the HeteroCache implementation directory.")
    p.add_argument("--load_dir", type=str, default="./data/clusters")
    p.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--method", type=str, default="HeteroCache",choices=["HeteroCache","SnapKV","H2O","StreamingLLM","PyramidKV","FullKV","CAKE"])

    p.add_argument("--compression_ratio", type=float, default=0.5, help="")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--quant", action="store_true",help="Enable 4bit quantization.")
    p.add_argument("--real_offload", action="store_true",help="Enable real offload.")
    p.add_argument("--decode_step", type=int, default=5000, help="")
    p.add_argument("--topk", type=int, default=1024, help="")
    p.add_argument("--stable_threshold", type=float, default=0.5)
    p.add_argument("--sim_threshold", type=float, default=0.5)
    args = p.parse_args()

    setup_heterocache_path(args.heterocache_path)

    model_name = args.model_name.lower()
    data_name = args.task

    model_path = args.model_path.lower()

    
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    args.max_gen_len = max_tokens
    from heterocache.utils import load_model_and_tokenizer
    model,tokenizer = load_model_and_tokenizer(args)


    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    # if 'sum' in args.task:
    #     GENERATION_CONFIG = GenerationConfig(
    #         temperature=0.8, top_p=0.95, max_new_tokens=max_tokens, do_sample=True, eos_token_id=tok.eos_token_id
    #     )
    # else:
    #     GENERATION_CONFIG = GenerationConfig(
    #         max_new_tokens=max_tokens, do_sample=False, eos_token_id=tok.eos_token_id
    #     )

    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}_{args.method}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}_{args.method}.jsonl"  # noqa
        )

    preds = []
    st = time.time()
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== {data_name} Example {i}/{args.stop_idx} ======")
        pred = get_pred(
            args, model, tokenizer, input_text, max_tokens=max_tokens, verbose=args.verbose
        )
        
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        print(f"#### std_answer {get_answer(eg, data_name)}")
        dump_jsonl(preds, output_path)
        torch.cuda.empty_cache()
    print("final total time", time.time() - st)
