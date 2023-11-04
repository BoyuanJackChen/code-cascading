from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import argparse
import torch
import yaml
from datasets import load_dataset
from langdetect import detect

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=int, default=1500)
FLAGS = parser.parse_args()

stop_words = ["\n\n", "\n\t\n"]
pe_py = "\n<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise in python:\n"

def trim_with_stopwords(generated_text, generated_list, stopwords, prompt, prompt_ids) -> str:
    answer_text = generated_text[len(prompt)-3:]
    answer_list = generated_list[len(prompt_ids):]
    min_i = len(answer_text)
    for w in sorted(stopwords):
        for i in range(len(answer_text)):
            if answer_text[i:].startswith(w) and min_i > i+len(w):
                min_i = i + len(w)
    answer_text = answer_text[:min_i]
    accumulator = "def"
    for i in range(len(answer_list)):
        accumulator += answer_list[i]
        if len(accumulator) >= len(answer_text):
            answer_list = answer_list[:i+1]
            break
    return answer_text, answer_list

def process_prompt(prompt):
    prompt = "Complete the Python function given the prompt below:\n" + prompt
    prompt = prompt.replace("\n\n\n", "\n")
    prompt = prompt.replace("\n\n", "\n")
    # prompt += pe_py
    prompt += "\n\ndef"
    return prompt

def is_english(text):
    try:
        language = detect(text)
    except:
        return False
    if language == 'en':
        return True
    else:
        return False

def main(args):
    # load yaml information
    model_index = args.file_name // 100
    if model_index == 3:
        model_size = "350M"
    elif model_index == 2:
        model_size = "2B"
    elif model_index == 6:
        model_size = "6B"
    elif model_index == 15:
        model_size = "starcoder"
    elif model_index == 16:
        model_size = "16B"
    start_question = args.file_name % 100 * 100
    end_question = start_question + 99

    number_key = "task_id"
    prompt_key = "prompt"
    difficulty_key = "difficulty"

    if model_size == "starcoder":
        output_file_name = f'starcoder.json'
    else:
        output_file_name = f'{model_size}.json'
    output_file_name = output_file_name.split(".json")[0] + f'/start{start_question}_end{end_question}.json'
    existing_data = []
    existing_numbers = []
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r') as f:
            existing_data = json.load(f)
        for data_dict in existing_data:
            existing_numbers.append(data_dict[number_key])

    # Initialize model and tokenizer
    if model_size == "starcoder":
        checkpoint = "bigcode/starcoder"
    else:
        checkpoint = f"Salesforce/codegen-{model_size}-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")

    # Load APPS dataset
    all_questions_dict = load_dataset("codeparrot/apps", split="test")

    # Generate answers
    number = -1
    answer_dict_list = []
    counter = 0
    output_data = []
    number_key = "problem_id"
    prompt_key = "question"
    if len(existing_data) > 0 and len(output_data) == 0:
        output_data = existing_data
        existing_data = []
    for question in all_questions_dict:
        number = question[number_key]
        if number in existing_numbers or number < start_question or number > end_question:
            print(f"Skipping question {number} because exists")
            continue
        prompt = question[prompt_key]
        if not is_english(prompt):
            print(f"Skipping question {number} because not English")
            continue
        if prompt.startswith("Write a function"):
            prompt = process_prompt(prompt)
        prompt = process_prompt(prompt)
        print(f"\nOn question {number}\n")
        difficulty = question[difficulty_key]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        generated_ids = model.generate(
            input_ids,
            use_cache=True,
            max_new_tokens=1024,
            num_beam_groups=4,
            num_beams=4,
            diversity_penalty=0.3,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        generated_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            generated_list.append(word)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        trimmed_text, trimmed_list = trim_with_stopwords(generated_text[0], generated_list, stop_words, prompt, prompt_ids)
        answer_dict = {"task_id": number, "prompt": prompt, "answer": trimmed_text, 
                       "difficulty": difficulty, "split": "test"}
        answer_dict_list.append(answer_dict)
        counter += 1

        # Update to json file
        if not os.path.exists(output_file_name):
            output_data = [answer_dict]
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=3)
            answer_dict_list = []
        elif counter >= 1:
            with open(output_file_name, 'r') as f:
                output_data = json.load(f)
            output_data += answer_dict_list
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=3)
            answer_dict_list = []


if __name__== "__main__":
    main(FLAGS)
