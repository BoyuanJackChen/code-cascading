from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
import time
import json
import os
import argparse
import torch
import random
import numpy as np
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=0, help="Model name")
parser.add_argument("--pass_at", type=int, default=1, help="pass @ how many")
parser.add_argument("--batch_size", type=int, default=180, help="Batch size for number of questions")
parser.add_argument("--num_loops", type=int, default=11, help="Number of times that we do this experiment")
FLAGS = parser.parse_args()

# We will hard-code the stop tokens for llama code family, as the tokenizer is automatically adding start tokens
stop_words = ["\n#", "\n```\n", "\n```\r", "\n```\n\n", ("\n```\n","\n")]
stop_words_ids = [[203,21], [203,914,203], [203,914,206], [203,914,553]]
assert_stop_words = ["assert"] + stop_words
assert_stop_words_ids = [[9294]] + stop_words_ids
eos_token_id = 0
eos_token = "<|endoftext|>"
imports = "\nimport math\nfrom typing import List\n"

def get_def_name(prompt):
    lines = prompt.split("\n")
    def_line = ""
    for line in reversed(lines):
        if line.startswith("def "):
            def_line = line
            break
    def_name = def_line.split(" ")[1].split("(")[0]
    return def_name

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    return answer

def trim_answer_from_start(answer):
    # Remove all beginning lines in answer, till it starts with "def ", "from" or "import"
    lines = answer.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("from") or line.startswith("import"):
            break
    answer = "\n".join(lines[i:])
    return answer

def process_answer(answer):
    answer = answer.replace("\r", "")
    answer = answer.replace("\t", "    ")
    answer = trim_answer_from_start(answer)
    answer = trim_substring_from_end(answer, "\n```\n")
    answer = trim_substring_from_end(answer, eos_token)
    answer = trim_substring_from_end(answer, "#")
    answer = trim_substring_from_end(answer, "print")
    answer = trim_substring_from_end(answer, "```")
    answer = trim_substring_from_end(answer, "\n\n")
    return answer

def process_test(answer, def_name):
    answer = answer.replace("\r", "")
    answer = answer.replace("\t", "    ")
    answer = trim_substring_from_end(answer, "assert")
    answer = trim_substring_from_end(answer, "if")
    answer = trim_substring_from_end(answer, "def")
    answer = trim_substring_from_end(answer, "print")
    answer = trim_substring_from_end(answer, eos_token)
    answer = trim_substring_from_end(answer, "\n```\n")
    answer = trim_substring_from_end(answer, "#")
    answer = trim_substring_from_end(answer, "```")
    answer = trim_substring_from_end(answer, "\n\n")
    answer = f"assert {def_name}" + answer
    return answer

def alpaca_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION

def custom_sample(range_size, batch_size):
    if batch_size <= range_size:
        # If batch_size is within the range, sample normally
        return random.sample(range(range_size), batch_size)
    else:
        # First, take all numbers in the range
        sample = list(range(range_size))
        # Then, randomly sample the remaining numbers needed to reach batch_size
        additional_samples = random.choices(range(range_size), k=batch_size - range_size)
        sample.extend(additional_samples)
        return sample

def custom_sample(range_size, batch_size):
    if batch_size <= range_size:
        # If batch_size is within the range, sample normally
        return random.sample(range(range_size), batch_size)
    else:
        # First, take all numbers in the range
        sample = list(range(range_size))
        # Then, randomly sample the remaining numbers needed to reach batch_size
        additional_samples = random.choices(range(range_size), k=batch_size - range_size)
        sample.extend(additional_samples)
        return sample


def main(args):
    loading_start = time.time()
    number_key = "task_id"
    prompt_key = "prompt"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    num_loops = args.num_loops
    pass_at = args.pass_at
    batch_size = args.batch_size
    all_time = np.zeros(num_loops)
    all_avg_cost = np.zeros(num_loops)
    all_numbers = list(range(4000, 5000))
    
    # Load APPS Dataset
    all_questions_dict = load_dataset("codeparrot/apps", split="test")
    number_key = "problem_id"
    prompt_key = "question"

    # Prepare the model checkpoint
    if args.model == 0:
        model_size = "1B"
    elif args.model == 1:
        model_size = "3B"
    elif args.model == 2:
        model_size = "15B"
    checkpoint = f"WizardLM/WizardCoder-{model_size}-V1.0"
    print(f"Humaneval; {checkpoint}")
    print(f"Pass @ {args.pass_at}")
    print(f"Batch size: {batch_size}")
    print(f"Num loops {args.num_loops}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
    loading_end = time.time()
    print(f"Time to load model is {loading_end - loading_start}")
    
    # Stopping criteria for generation using the LogitsProcessor class
    class StopSequences(LogitsProcessor):
        def __init__(self, stop_ids, batch_size, encounters=1, eos_token_id=eos_token_id):
            StoppingCriteria.__init__(self)
            self.stop_sequences = stop_ids
            self.batch_size = batch_size
            self.encounters = [encounters] * batch_size
            self.NUM_ENCOUNTERS = encounters
            self.eos_token_id = eos_token_id

        def __call__(self, input_ids, scores):
            forced_eos = torch.full((scores.size(1),), -float("inf"))
            forced_eos[self.eos_token_id] = 0
            for stop in self.stop_sequences:
                # Check if the input_ids end with the stop sequence
                for i in range(self.batch_size):
                    if self.encounters[i] <= 0:
                        continue
                    if input_ids[i][-len(stop):].tolist() == stop:
                        self.encounters[i] -= 1
                        if self.encounters[i] <= 0:
                            scores[i] = forced_eos
            return scores

    for loop in range(num_loops):
        all_answer_prompts = []
        selected_numbers = random.sample(all_numbers, batch_size)
        # print(selected_numbers)
        for number in selected_numbers:
            question = all_questions_dict[number]
            prompt = question[prompt_key]
            prompt = prompt.replace('    ', '\t')
            question_prompt = alpaca_prompt(prompt)
            all_answer_prompts += [question_prompt]*max(pass_at,1)
        logits_processor = LogitsProcessorList([StopSequences(stop_words_ids, batch_size=batch_size*max(pass_at,1), encounters=1)])
        
        # Generate answers
        start = time.time()
        prompt_ids = tokenizer.batch_encode_plus(
                        all_answer_prompts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=2048
                    ).to(torch.cuda.current_device())
        max_new_tokens = 1024
        max_length = 2048
        with torch.no_grad():
            answer_ids = model.generate(
                **prompt_ids,
                use_cache = True,
                pad_token_id = tokenizer.eos_token_id,
                eos_token_id = tokenizer.eos_token_id,
                max_new_tokens = max_new_tokens,
                do_sample = True,
                top_k = 0,
                top_p = 0.95,
                temperature = 0.8,
                num_beams = 1,
                logits_processor = logits_processor
            )
        answer_ids = answer_ids[:, len(prompt_ids['input_ids'][0]):]
        answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
        # answer_trimmed = [process_answer(answer) for answer in answer_text]
        torch.cuda.empty_cache()
        end = time.time()
        time_spent = round(end-start, 2)
        all_time[loop] = time_spent
        total_count = sum(tensor.ne(eos_token_id).sum().item() for tensor in answer_ids)
        print(f"Loop {loop} time spent: {time_spent} seconds; num tokens: {total_count}")
        time_per_1k_tokens = round(time_spent / (total_count / 1000), 2)
        all_avg_cost[loop] = time_per_1k_tokens
        print(f"Time per 1k tokens: {time_per_1k_tokens} seconds")

    print(f"Average time per 1k tokens: {np.mean(all_avg_cost[1:])} seconds")

if __name__== "__main__":
    main(FLAGS)
