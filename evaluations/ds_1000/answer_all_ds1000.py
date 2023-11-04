from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import argparse
import yaml
from datasets import load_dataset
from langdetect import detect
import torch

from ds1000 import DS1000Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=int, default=3500)
FLAGS = parser.parse_args()

stop_words = ["\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n"]

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    trimmed = False
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt-3:]   # Including 'def' at the beginning
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
                    trimmed = True
        answer = answer[:min_i]
        result.append(answer)
    if not trimmed:
        print("\nThis question is not trimmed!\n")
    return result

def process_prompt(prompt):
    prompt = "Complete the Python function given the prompt below:" + prompt
    prompt = prompt.replace("\n\n\n", "\n")
    prompt = prompt.replace("\n\n", "\n")
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
    model_index = args.file_name // 100
    library_key = args.file_name % 100
    if model_index == 0:
        model_size = "starcoder"
    elif model_index == 35:
        model_size = "350M"
    elif model_index == 2:
        model_size = "2B"
    elif model_index == 6:
        model_size = "6B"
    elif model_index == 16:
        model_size = "16B"
    
    if model_index == 0:
        output_file_name = f'starcoder.json'
    else:
        output_file_name = f'{model_size}.json'

    # Load dataset
    number_key = "perturbation_origin_id"
    prompt_key = "prompt"

    # Initialize model and tokenizer
    if model_size == "starcoder":
        checkpoint = "bigcode/starcoder"
    else:
        checkpoint = f"Salesforce/codegen-{model_size}-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")

    # Load dataset
    ds_data = DS1000Dataset("ds1000_data")

    # Generate answers
    answer_dict_list = []
    counter = 0

    # for key in ['Matplotlib', 'Numpy', 'Pandas', 'Pytorch', 'Scipy', 'Sklearn', 'Tensorflow']:
    all_libraries = ['Matplotlib', 'Numpy', 'Pandas', 'Pytorch', 'Scipy', 'Sklearn', 'Tensorflow']
    key = all_libraries[library_key]
    if not os.path.exists(key):
        os.makedirs(key)
    output_file_name = key + "/" + output_file_name
    output_data = []
    all_questions_dict = ds_data[key]
    if os.path.exists(output_file_name):
        # Load the json file output_file_name
        with open(output_file_name, 'r') as f:
            existing_data = json.load(f)
        existing_len = len(existing_data)
    else:
        existing_len = 0
        
    for i in range(existing_len, len(all_questions_dict)):
        question = all_questions_dict[i]
        number = int(question[number_key])
        prompt = question[prompt_key]
        if not is_english(prompt):
            print(f"Skipping question {number} because not English")
            continue
        prompt = prompt.replace("\t", "    ")
        prompt = process_prompt(prompt)
        print(f"\nOn question {number}\n")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if len(input_ids) > 2048 and model_size != "starcoder":
            print(f"Skipping question {number} in {key}, because too long")
            continue
        generated_ids = model.generate(
            input_ids,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=500,
            num_beam_groups=4,
            diversity_penalty=0.3,
            num_beams=4,
            do_sample=False,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt = tokenizer.decode(prompt_ids[0])
        trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
        answer_dict = {"library_id": i, "source_id": number, 
                       "prompt": prompt, "answer": trimmed_text[0]}
        answer_dict_list.append(answer_dict)
        counter += 1

        # Update to json file
        if not os.path.exists(output_file_name):
            output_data = [answer_dict]
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
            answer_dict_list = []
        elif counter >= 1:
            with open(output_file_name, 'r') as f:
                output_data = json.load(f)
            output_data += answer_dict_list
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
            answer_dict_list = []


if __name__== "__main__":
    main(FLAGS)
