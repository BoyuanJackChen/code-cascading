from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import argparse
import yaml
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="get_all.yaml")
FLAGS = parser.parse_args()

eos_token = 50256
stop_words = ["\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n"]
instruction = "Complete the Python function given the prompt below. Do not copy the definition line, just start" \
              "with the body of the function.\n"

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    trimmed = False
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
                    trimmed = True
        answer = answer[:min_i]
        result.append(answer)
    if not trimmed:
        print("This question is not trimmed!")
    return result


def main(args):
    # load yaml information
    with open(args.file_name) as f:
        model_params = yaml.load(f, Loader=yaml.FullLoader)
    k = model_params["k"]
    model_size = model_params["model_size"]
    start_question = 0
    end_question = 10000
    if "start" in model_params:
        start_question = model_params["start"]
    if "end" in model_params:
        end_question = model_params["end"]

    number_key = "task_id"
    title_key = "entry_point"
    prompt_key = "prompt"
    test_key = "test"

    # Set output file; track and trim already generated data. Currently this is only for k=1,
    # because k>1 takes forever and I need to figure this out with Marco.
    if model_size == "starcoder":
        output_file_name = f'starcoder_k{k}.json'
        eos_token = 0
    else:
        output_file_name = f'{model_size}_k{k}.json'
    if "start" in model_params:
        output_file_name = output_file_name.split(".json")[0] + f'_start{start_question}_end{end_question}.json'
    existing_data = []
    existing_numbers = []
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r') as f:
            existing_data = json.load(f)
        for data_dict in existing_data:
            existing_numbers.append(data_dict["number"])
        if k > 1:
            the_incomplete_num = -1
            current_num = -1
            max_pass = -1
            for data_dict in existing_data:
                if data_dict[number_key] != current_num:
                    if max_pass < k:
                        the_incomplete_num = current_num
                        continue
                    existing_numbers.append(current_num)
                    current_num = data_dict[number_key]
                    max_pass = 0
                else:
                    max_pass += 1
            print(f"The incomplete number is {the_incomplete_num}")
            print(f"existing_numbers is {existing_numbers}")
            for data_dict in existing_data:
                if data_dict[number_key] == the_incomplete_num:
                    existing_data.remove(data_dict)

    # Initialize model and tokenizer
    if model_size == "starcoder":
        checkpoint = "bigcode/starcoder"
    else:
        checkpoint = f"Salesforce/codegen-{model_size}-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    model.eval()
    print(f"Time to load model is {time.time() - start_load_model}")

    # Load Leetcode Dataset
    all_questions_dict = json.load(open("../data/HumanEval.json", "r"))

    # Generate answers
    answer_dict_list = []
    counter = 0

    start = time.time()
    for question in all_questions_dict:
        number = question[number_key]
        if number in existing_numbers:
            print(f"Skipping question {number}")
            continue
        title = question[title_key]
        prompt = question[prompt_key]
        prompt = prompt.replace("\t", "    ")
        print(f"On question {number}: {title}")
        if question[test_key].startswith("\n\nMeta"):
            test_list = question[test_key].split("\n\n\n")[1]
        else:
            test_list = question[test_key]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"input ids is {input_ids} has length {len(input_ids)}")
        if model_size == "starcoder":
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            generated_ids = model.generate(
                input_ids,
                max_length=500,
                num_beams=4,
                num_beam_groups=4,
                diversity_penalty=0.3,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"generated ids is {generated_ids[0]} has length {len(generated_ids[0])}")
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        print(f"generated text is {generated_text[0]} has length {len(generated_text[0])}")
        
        answer_dict = {"number": number, "title": title, "answer": generated_text[0],
                        "test": test_list}
        answer_dict_list.append(answer_dict)
        print(generated_text[0])
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

    end = time.time()
    print(f"Time to generate HumanEval for 16B with full capacity is {end - start} seconds")

if __name__== "__main__":
    main(FLAGS)
