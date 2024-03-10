from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
import time
import json
import os
import argparse
import multiprocessing
import torch
from human_eval.data import write_jsonl, read_problems, stream_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=0, help="Model name")
parser.add_argument("--pass_at", type=int, default=1, help="pass @ how many")
FLAGS = parser.parse_args()

# We will hard-code the stop tokens for llama code family, as the tokenizer is automatically adding start tokens
stop_words = ["\n#", "\n```\n", "\n```\r", "\n```\n\n", ("\n```\n","\n")]
stop_words_ids = [[203,21], [203,914,203], [203,914,206], [203,914,553]]
assert_stop_words = ["assert"] + stop_words
assert_stop_words_ids = [[9294]] + stop_words_ids
eos_token_id = 0
eos_token = "<|endoftext|>"
imports = "\nimport math\nfrom typing import List\n"
fixed_starter = """Here's the Python script for the problem:

```python
"""
fixed_starter_ids = [10921, 1182, 322, 4865, 3261, 436, 322, 3708, 44, 553, 203, 914, 2958, 206, 203]

single_replacements = {
    478: 553,
    284: 737,
    2448: [737, 2448],  # Now reflects one value to two directly
}
consecutive_replacements = {
    (16843, 3246): 39394,
}

def replace_tensor_values(tensor, single_replacements, consecutive_replacements):
    # First, handle consecutive replacements to avoid breaking sequences with single replacements
    tensor_list = tensor.tolist()
    result_list = []

    i = 0
    while i < len(tensor_list):
        matched = False
        for old_values, new_value in consecutive_replacements.items():
            # Check if the current sequence matches any of the old_values sequences
            if tensor_list[i:i+len(old_values)] == list(old_values):
                # If matched, replace with new_value
                result_list.append(new_value)
                i += len(old_values)
                matched = True
                break
        if not matched:
            # If no match, just move the current value to result_list and handle possible single replacements later
            result_list.append(tensor_list[i])
            i += 1

    # Now, handle single replacements on the result_list
    final_list = []
    for value in result_list:
        if value in single_replacements:
            replacement = single_replacements[value]
            if isinstance(replacement, list):
                final_list.extend(replacement)  # Extend if the replacement is a list of values
            else:
                final_list.append(replacement)  # Append if it's a single value
        else:
            final_list.append(value)  # Append the original value if no replacement found

    return torch.tensor(final_list, dtype=tensor.dtype)

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
    # answer = answer[:answer.find("\n#")]
    answer = answer.replace("\r", "")
    answer = answer.replace("\t", "    ")
    # answer = trim_answer_from_start(answer)
    answer = trim_substring_from_end(answer, eos_token)
    answer = trim_substring_from_end(answer, "\n```\n")
    answer = trim_substring_from_end(answer, "#")
    answer = trim_substring_from_end(answer, "```")
    answer = trim_substring_from_end(answer, "\n\n")
    return answer

def alpaca_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION

def move_values_to_front(tensor, value):
    mask = (tensor == value)
    count = torch.sum(mask).item()
    new_tensor = torch.cat([torch.full((count,), value, dtype=tensor.dtype).to(tensor.device), tensor[tensor != value]])
    return new_tensor

def alpaca_hardcode(prompts, fixed_starter_ls, tokenizer):
    # Get the alpaca prompt ids
    all_alpaca = []
    for prompt in prompts:
        all_alpaca.append(alpaca_prompt(prompt))
    input_ids = tokenizer.batch_encode_plus(all_alpaca, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(torch.cuda.current_device())
    
    # Append the fixed starters
    fixed_starter_ids = torch.tensor([fixed_starter_ls]*len(prompts)).to(torch.cuda.current_device())
    fixed_starter_ones = torch.ones(fixed_starter_ids.shape, dtype=torch.int64).to(torch.cuda.current_device())
    input_ids["input_ids"] = torch.cat((input_ids["input_ids"], fixed_starter_ids), dim=1)
    input_ids["attention_mask"] = torch.cat((input_ids["attention_mask"], fixed_starter_ones), dim=1)

    # Append the prompts again
    prompts_ids = tokenizer.batch_encode_plus(prompts, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(torch.cuda.current_device())
    input_ids["input_ids"] = torch.cat((input_ids["input_ids"], prompts_ids["input_ids"]), dim=1)
    input_ids["attention_mask"] = torch.cat((input_ids["attention_mask"], prompts_ids["attention_mask"]), dim=1)
    for i in range(input_ids["input_ids"].shape[0]):
        input_ids["input_ids"][i] = move_values_to_front(input_ids["input_ids"][i], tokenizer.pad_token_id)
        target_input_ids = replace_tensor_values(input_ids["input_ids"][i], single_replacements, consecutive_replacements)
        # input_ids["input_ids"] = torch.cat((input_ids["input_ids"], torch.ones((1,2), dtype=torch.int64).to(input_ids["input_ids"].device)), dim=1)
        input_ids["input_ids"][i] = target_input_ids
        input_ids["attention_mask"][i] = move_values_to_front(input_ids["attention_mask"][i], 0)
        # input_ids["attention_mask"] = torch.cat((input_ids["attention_mask"], torch.ones((1,2), dtype=torch.int64).to(input_ids["attention_mask"].device)), dim=1)
    
    return input_ids

def alpaca_test(input):
    lines = input.split("\n")
    def_line = ""
    for line in reversed(lines):
        if line.startswith("def "):
            def_line = line
            break
    def_name = def_line.split(" ")[1].split("(")[0]
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a set of tests for this function:
{input}

### Response:
assert {def_name}("""
    return INSTRUCTION


def main(args):
    loading_start = time.time()
    number_key = "task_id"
    prompt_key = "prompt"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pass_at = args.pass_at

    # Prepare the model checkpoint (just 1)
    answer_dict_list = []
    counter = 0
    if args.model == 0:
        model_size = "1B"
    elif args.model == 1:
        model_size = "3B"
    elif args.model == 2:
        model_size = "15B"
    checkpoint = f"WizardLM/WizardCoder-{model_size}-V1.0"
    print(f"Model is {checkpoint}")
    print(f"Pass @ {args.pass_at}")
    
    # Load HumanEval Dataset
    all_questions_dict = read_problems()
    all_keys = ["HumanEval/3", "HumanEval/2", "HumanEval/0", "HumanEval/8", "HumanEval/145"]
    all_keys = ["HumanEval/3"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    all_prompts = [all_questions_dict[ak]["prompt"] for ak in all_keys]
    prompt_ids = alpaca_hardcode(all_prompts, fixed_starter_ids, tokenizer)
    print(prompt_ids)
    print(tokenizer.decode(prompt_ids["input_ids"][0]))
    input()
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto")
    model.eval()
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
            self.just_started = True

        def __call__(self, input_ids, scores):
            forced_eos = torch.full((scores.size(1),), -float("inf"))
            forced_eos[self.eos_token_id] = 0
            for stop in self.stop_sequences:
                # Check if the input_ids end with the stop sequence
                for i in range(self.batch_size):
                    if self.encounters[i] <= 0:
                        continue
                    if input_ids[i][-len(stop):].tolist() == stop:
                        if self.just_started:
                            self.just_started = False
                            continue
                        self.encounters[i] -= 1
                        if self.encounters[i] <= 0:
                            scores[i] = forced_eos
            return scores


    logits_processor = LogitsProcessorList([StopSequences(stop_words_ids, batch_size=prompt_ids["input_ids"].shape[0], encounters=1)])
    
    # Generate answers
    print(f"Generation started...")
    start = time.time()
    max_new_tokens = 1024
    with torch.no_grad():
        if pass_at in [0,1]:
            answer_ids = model.generate(
                **prompt_ids,
                use_cache = True,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
                max_new_tokens = max_new_tokens,
                num_return_sequences=1,
                do_sample = False,
                logits_processor = logits_processor
            )
        else:
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
    print(answer_ids)
    num_tokens = answer_ids.size(1)
    answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
    answer_trimmed = [process_answer(answer) for answer in answer_text]
    torch.cuda.empty_cache()
    for at in answer_trimmed:
        print(at)
        print("\n------\n")
    print(f"Time to generate is {time.time() - start} seconds")
    print(f"Per-token time is {(time.time() - start)/num_tokens} seconds")
        
                

if __name__== "__main__":
    main(FLAGS)
