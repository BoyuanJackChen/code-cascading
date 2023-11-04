from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="get_all.yaml")
FLAGS = parser.parse_args()

stop_words = ["\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n"]

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
        print("\nThis question is not trimmed!\n")
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

    # Set output file; track and trim already generated data. Currently this is only for k=1,
    # because k>1 takes forever and I need to figure this out with Marco.
    if model_size == "starcoder":
        output_file_name = f'starcoder_k{k}.json'
    else:
        output_file_name = f'{model_size}_k{k}.json'
    if "start" in model_params:
        output_file_name = f'{model_size}_k{k}_start{start_question}_end{end_question}.json'
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
                if data_dict["number"] != current_num:
                    if max_pass < k:
                        the_incomplete_num = current_num
                        continue
                    existing_numbers.append(current_num)
                    current_num = data_dict["number"]
                    max_pass = 0
                else:
                    max_pass += 1
            print(f"The incomplete number is {the_incomplete_num}")
            print(f"existing_numbers is {existing_numbers}")
            for data_dict in existing_data:
                if data_dict["number"] == the_incomplete_num:
                    existing_data.remove(data_dict)

    # Initialize model and tokenizer
    if model_size == "starcoder":
        checkpoint = "bigcode/starcoder"
    else:
        checkpoint = f"Salesforce/codegen-{model_size}-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")

    # Load Leetcode Dataset
    all_questions_dict = json.load(open("../parser/data.json", "r"))

    # Generate answers
    number = -1
    start = time.time()
    answer_dict_list = []
    counter = 0

    # When we need multiple answers from the same question, we use sampling generate method
    if k > 1:
        for question in all_questions_dict:
            if question["number"] in existing_numbers:
                continue
            for i in range(k):
                if number != question["number"]:
                    print(f"Question {number} took {time.time() - start} to generate {k} answers")
                    start = time.time()
                    number = question["number"]
                title = question["title"]
                difficulty = question["difficulty"]
                prompt = question["question"]
                prompt = prompt.replace("\t", "    ")
                print(f"\nOn question {number}: {title} (Difficulty {difficulty})")

                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=250,
                    num_beams=4,
                    num_beam_groups=1,
                    do_sample=True,
                )
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                decoded_list = []
                for ids in generated_ids[0]:
                    word = tokenizer.decode(int(ids))
                    decoded_list.append(word)
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
                prompt = tokenizer.decode(prompt_ids[0])
                trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
                answer_dict = {"number": number, "title": title, "difficulty": difficulty,
                               "pass": i, "answer": trimmed_text[0]}
                answer_dict_list.append(answer_dict)
                counter += 1

                # Update to json file
                if not os.path.exists(output_file_name):
                    output_data = [answer_dict]
                    with open(output_file_name, 'w') as f:
                        json.dump(output_data, f, indent=4)
                elif counter >= k:
                    if len(existing_data) == 0:
                        with open(output_file_name, 'r') as f:
                            output_data = json.load(f)
                    else:
                        output_data = existing_data
                        existing_data = []
                    output_data += answer_dict_list
                    with open(output_file_name, 'w') as f:
                        json.dump(output_data, f, indent=4)
                    answer_dict_list = []
                    counter = 0

    else:   # k=1
        output_data = []
        if len(existing_data) > 0 and len(output_data) == 0:
            output_data = existing_data
            existing_data = []
        for question in all_questions_dict:
            number = question["number"]
            if number in existing_numbers or number < start_question or number >= end_question:
                print(f"Skipping question {number}")
                continue
            title = question["title"]
            difficulty = question["difficulty"]
            prompt = question["question"]
            prompt = prompt.replace("\t", "    ")
            print(f"\nOn question {number}: {title} (Difficulty {difficulty})")

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=250,
                num_beam_groups=4,
                diversity_penalty=0.3,
                num_beams=4,
                use_cache=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            decoded_list = []
            for ids in generated_ids[0]:
                word = tokenizer.decode(int(ids))
                decoded_list.append(word)
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
            prompt = tokenizer.decode(prompt_ids[0])
            trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
            answer_dict = {"number": number, "title": title, "difficulty": difficulty, "answer": trimmed_text[0]}
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
