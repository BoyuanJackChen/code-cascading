from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=int, default=0)
FLAGS = parser.parse_args()

eos_token = 50256
stop_words = ["<|endoftext|>", "\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n"]

def trim_with_stopwords(generated_text, generated_list, stopwords, prompt, prompt_ids) -> str:
    original_prompt = prompt.rsplit("\n\n", 1)[0] + "\n\n"
    answer_text = generated_text[len(original_prompt):]
    answer_list = generated_list[len(prompt_ids):]
    min_i = len(answer_text)
    for w in sorted(stopwords):
        for i in range(len(answer_text)):
            if answer_text[i:].startswith(w) and min_i > i+len(w):
                min_i = i + len(w)
    answer_text = answer_text[:min_i]
    accumulator = ""
    lines = answer_text.splitlines()
    for line in lines:
        if line.startswith("def"):
            accumulator = line
            break
    for i in range(len(answer_list)):
        accumulator += answer_list[i]
        if len(accumulator) >= len(answer_text):
            answer_list = answer_list[:i+1]
            break
    return answer_text, answer_list


def process_prompt(prompt, canonical_solution):
    def_line = ""
    lines = canonical_solution.splitlines()
    for line in lines:
        if line.startswith("def"):
            def_line = line
    def_line = def_line.replace(" \r", "")
    if prompt.startswith("Write a function"):
        prompt = prompt.replace("Write a function", "Write a python function", 1)
    prompt += "\n\n" + def_line + "\n"
    return prompt


def main(args):
    # load yaml information
    model_number = args.file_name
    model_index = model_number // 100
    if model_index == 0:
        output_file_name = f'starcoder.json'
        checkpoint = "bigcode/starcoder"
    elif model_index == 35:
        output_file_name = f'350M.json'
        checkpoint = f"Salesforce/codegen-350M-mono"
    elif model_index == 2:
        output_file_name = f'2B.json'
        checkpoint = f"Salesforce/codegen-2B-mono"
    elif model_index == 6:
        output_file_name = f'6B.json'
        checkpoint = f"Salesforce/codegen-6B-mono"
    elif model_index == 16:
        output_file_name = f'16B.json'
        checkpoint = f"Salesforce/codegen-16B-mono"
    else:
        print("Invalid model number")
    existing_data = []
    existing_numbers = []
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r') as f:
            existing_data = json.load(f)
        for data_dict in existing_data:
            existing_numbers.append(data_dict[number_key])
    start_question = 0
    end_question = 1000
            
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")

    # Load MBPP dataset
    with open('../mbpp.jsonl', 'r') as f:
        all_questions_dict = []
        for line in f:
            all_questions_dict.append(json.loads(line))
    prompt_key = "text"
    number_key = "task_id"

    # Generate answers
    number = -1
    answer_dict_list = []
    counter = 0

    output_data = []
    if len(existing_data) > 0 and len(output_data) == 0:
        output_data = existing_data
        existing_data = []
    for question in all_questions_dict:
        number = question[number_key]
        test_list = question["test_list"]
        canonical_solution = question["code"]
        if number in existing_numbers or number < start_question or number >= end_question:
            print(f"Skipping question {number}")
            continue
        prompt = question[prompt_key]
        prompt = process_prompt(prompt, canonical_solution)
        print(f"\nOn question {number}: {prompt}")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            max_new_tokens=1024,
            num_beams=4,
            num_beam_groups=4,
            diversity_penalty=0.3,
            do_sample=False
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        generated_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            generated_list.append(word)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt = tokenizer.decode(prompt_ids)
        trimmed_text, trimmed_list = trim_with_stopwords(generated_text[0], generated_list, stop_words, prompt, prompt_ids)
        answer_dict = {"task_id": number, "prompt": prompt, "answer": trimmed_text, 
                       "test_list": test_list, "answer_list": trimmed_list}
        answer_dict_list.append(answer_dict)
        counter += 1

        # Update to json file
        if not os.path.exists(output_file_name):
            output_data = [answer_dict]
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
        elif counter >= 1:
            with open(output_file_name, 'r') as f:
                output_data = json.load(f)
            output_data += answer_dict_list
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
            answer_dict_list = []
            counter = 0


if __name__== "__main__":
    main(FLAGS)
