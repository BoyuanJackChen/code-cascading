from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from human_eval.data import read_problems
from tqdm import tqdm
import os, sys
import json

# Add the current directory to the beginning of sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = current_dir[:current_dir.rfind('/')]
sys.path.insert(0, current_dir)
from bsp.generator import SpeculativeGenerationModel

fixed_starter = "Here's the Python script for the given problem:\n\n```python\n"
fixed_starter_ids_sc = [10921, 1182, 322, 4865, 3261, 436, 322, 3708, 44, 553, 203, 914, 2958, 206, 203]
def alpaca_prompt(input, apps=False):
    if apps:
        INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:
def solution(stdin: str) -> str:"""
    else:
        INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION

@torch.inference_mode()
def generate_hf(prompts, model, tokenizer, step):
    tokenizer.padding_side='left'
    gen_conf = GenerationConfig(max_new_tokens=step, min_new_tokens=step, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    token_seqs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(**token_seqs, generation_config=gen_conf)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

@torch.inference_mode()
def generate_hf_assist(prompts, model, assist_model, tokenizer, step):
    tokenizer.padding_side='left'
    gen_conf = GenerationConfig(max_new_tokens=step, min_new_tokens=step, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    ret = []
    for p in prompts:
        token_seqs = tokenizer(p, return_tensors="pt")
        token_seqs = token_seqs.to('cuda')
        model = model.to('cuda')
        out = model.generate(**token_seqs, generation_config=gen_conf, assistant_model=assist_model)
        ret.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return ret

def get_dataset(dataset_name, truncate=None, alpaca=True):
    dataset = []
    if dataset_name == 'alespalla/chatbot_instruction_prompts':
        dataset = load_dataset(dataset_name)
        dataset = [t['prompt'] for t in dataset['test']]
    elif dataset_name == 'human-eval' or dataset_name == 'HumanEval' or dataset_name == 'humaneval':
        dataset = read_problems()
        prompt_dataset = []
        for k in range(0,164):
            original_prompt = dataset[f"HumanEval/{k}"]['prompt']
            original_prompt = original_prompt.replace('    ', '\t')
            prompt = alpaca_prompt(original_prompt) if alpaca else prompt
            prompt_dataset.append(prompt)
        dataset = prompt_dataset
    elif dataset_name == 'mbpp' or dataset_name == 'MBPP':
        data_file = "../evaluations/mbpp/mbpp_sanitized_for_code_generation_codet.jsonl"
        dataset = []
        with open(data_file, 'r') as file:
            for line in file:
                json_line = json.loads(line)
                prompt = alpaca_prompt(prompt, apps=False) if alpaca else json_line
                dataset.append(json_line)
    elif dataset_name == 'apps-intro' or dataset_name == 'apps_intro':
        all_questions_dict = load_dataset("codeparrot/apps", split="test")
        number_key = "problem_id"
        prompt_key = "question"
        dataset = []
        for qd in all_questions_dict:
            number = qd[number_key]
            if number < 4000:
                continue
            prompt = qd[prompt_key]
            prompt = prompt.replace('    ', '\t')
            prompt = alpaca_prompt(prompt, apps=True) if alpaca else prompt
            dataset.append(prompt)
    else:
        raise ValueError("Unsupported dataset")
    if truncate is not None:
        dataset = dataset[:truncate]
        return dataset
    return dataset

def benchmark(gen_fn, prompts, batch_size, warmup=3):
    for w in range(warmup):
        print(f"Doing wampup {w+1}/{warmup}")
        out = gen_fn(prompts[:batch_size])
    data_loader = DataLoader(prompts, batch_size=batch_size, shuffle=True)
    generated_seqs = []
    torch.cuda.synchronize()
    start_t = time.time()
    valid_token_count = 0
    for prompt in tqdm(data_loader):
        batch_start = time.time()
        result, vtc = gen_fn(prompt)
        generated_seqs.extend(result)
        valid_token_count += vtc
        batch_end = time.time()
        # for r in result:
        #     print(f"{r}")
        # print(f"Batch time: {batch_end - batch_start}; valid token count: {vtc}; per token time: {(batch_end - batch_start) / vtc}s")
    torch.cuda.synchronize()
    dur = time.time() - start_t
    return dur, generated_seqs, valid_token_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--assist-model', type=str)
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--speculate-steps', type=int, nargs='+')
    parser.add_argument('--len-out', type=int)
    parser.add_argument('--batch-sizes', type=int, nargs='+')
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--collect-stats', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset-truncate', type=int)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    print(args)

    # load dataset
    alpaca = "Wizard" in args.model
    prompts = get_dataset(args.dataset, args.dataset_truncate, alpaca)

    # Initialized the two models
    print(f"Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto")
    model.eval()
    assist_model = AutoModelForCausalLM.from_pretrained(
        args.assist_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto")
    assist_model.eval()

    if "Wizard" in args.model and "Python" not in args.model:
        if "apps" in args.dataset:
            stopping_ids = [[203,21], [203,914,203], [203,914,206], [203,914,553]]
        else:
            stopping_ids = [[203,21], [203,914,203], [203,914,206], [203,914,553]]
    elif "Wizard" in args.model and "Python" in args.model:
        if "apps" in args.dataset:
            stopping_ids = [[13,29937], [13,28956,13], [13,28956,30004], [13,361], [13,1753]]  # \ndef
        else:
            stopping_ids = [[13,29937], [13,28956,13], [13,28956,30004], [13,361]]
    # elif "Codegen" in args.model:
        

    print(f"All batch sizes: {args.batch_sizes}; all speculate steps: {args.speculate_steps}. Now generating...")
    for batch_size in args.batch_sizes:
        for speculate_step in args.speculate_steps:
            assist_model.max_assistant_tokens = speculate_step
            generator = SpeculativeGenerationModel(model, assist_model, tokenizer, speculate_step)
            if speculate_step == 0:
                t, ret = benchmark(lambda p: generate_hf(p, model, tokenizer, args.len_out), prompts, batch_size, warmup=args.warmup)
            else:
                t, ret, valid_token_count = benchmark(lambda p: generator.generate(p, args.len_out, collect_stats=args.collect_stats, stopping_ids=stopping_ids), prompts, batch_size, warmup=args.warmup)
            # num_tokens = len(ret) * args.len_out
            num_tokens = valid_token_count
            print(f"\nBatch size: {batch_size}, Spec step: {speculate_step}, total time: {t}s, valid token num: {valid_token_count}; Time per token: {t / num_tokens}")
            for answer in ret:
                print(answer)
                print("\n-----------------------------\n")
            
            if args.collect_stats:
                hit_rate, time_speculate, time_verify, verify_calls = generator.get_stats()
                print("speculation hit rate:", ', '.join([str(h.cpu().numpy()) for h in hit_rate]))
                print("expected correct speculated length:", hit_rate.sum())
                print(f"time for speculation {time_speculate} s | verification {time_verify} s | #verifys: {verify_calls}")
                print(f"\nBatch size: {batch_size}, Spec step: {speculate_step}, total time: {t}s, valid token num: {valid_token_count}; Time per token: {t / num_tokens}")