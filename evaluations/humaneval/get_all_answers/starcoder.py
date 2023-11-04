from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import argparse
import yaml
import torch

eos_token = 50256
stop_words = ["\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n"]
pe = "<filename>solutions/solution_1.py# Here is the correct implementation of the code exercise\n"


def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    trimmed = False
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt*2-10:]
        for i in range(len(answer)):
            if answer[i:].startswith('"""\n'):
                answer = answer[i+4:]
                break
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


def main():
    k = 1
    model_size = "starcoder"
    output_file_name = f'starcoder_k{k}.json'
    start_question = 0
    end_question = 164

    number_key = "task_id"
    title_key = "entry_point"
    prompt_key = "prompt"
    test_key = "test"
    
    existing_data = []
    existing_numbers = []
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r') as f:
            existing_data = json.load(f)
        for data_dict in existing_data:
            existing_numbers.append(data_dict["number"])

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

    for question in all_questions_dict:
        number = question[number_key]
        if number in existing_numbers:
            print(f"Skipping question {number}")
            continue
        title = question[title_key]
        prompt = question[prompt_key]
        prompt = prompt.replace("\t", "    ")
        # A very important line!
        prompt += pe
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
                num_beams=4,
                num_beam_groups=4,
                diversity_penalty=0.3,
                # do_sample=True,
                # temperature=0.5,
                # pad_token_id=tokenizer.eos_token_id,
            )
        else:
            generated_ids = model.generate(
                input_ids,
                max_length=400,
                num_beams=4,
                num_beam_groups=4,
                diversity_penalty=0.3,
            )
        # print(f"generated ids is {generated_ids[0]} has length {len(generated_ids[0])}")
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        print(f"generated text is {generated_text[0]} has length {len(generated_text[0])}")
        if model_size != "starcoder":
            decoded_list = []
            for ids in generated_ids[0]:
                word = tokenizer.decode(int(ids))
                decoded_list.append(word)
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
            prompt = tokenizer.decode(prompt_ids[0])
            trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
            answer_dict = {"number": number, "title": title, "answer": trimmed_text[0],
                           "test": test_list}
            answer_dict_list.append(answer_dict)
        else:
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


if __name__== "__main__":
    main()
