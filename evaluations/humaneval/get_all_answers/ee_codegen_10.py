from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import torch
from model_classifier import MLPClassifier, LSTMClassifier
import accelerate

accelerator = accelerate.Accelerator()
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
        print("This question is not trimmed!")
    return result


def main():
    k = 1
    model_size = "16B"
    number_key = "task_id"
    title_key = "entry_point"
    prompt_key = "prompt"
    test_key = "test"

    # w5/epoch18: 0.99 saves 4.0 layers with full accuracy
    # w10/epoch27: 0.995 saves 4.2 layers with full accuracy
    output_file_name = f'16B_ee_w10_0.995.json'
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

    ee_model = LSTMClassifier()
    ee_model = torch.load("../../../ee_classifier/checkpoints_saved/w10/epoch_27.pt")
    ee_model = accelerator.prepare(ee_model)
    ee_model.eval()
    print(f"Time to load model is {time.time() - start_load_model}")

    model.set_ee_sequential(ee_model)
    model.set_ee_threshold(0.995)

    # Load Leetcode Dataset
    all_questions_dict = json.load(open("../data/HumanEval.json", "r"))

    # Generate answers
    answer_dict_list = []
    counter = 0

    start = time.time()
    for question in all_questions_dict:
        number = question[number_key]
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
                num_beams=4,
                num_beam_groups=4,
                diversity_penalty=0.3,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            generated_ids = model.generate(
                input_ids,
                max_length=400,
                num_beams=4,
                num_beam_groups=4,
                diversity_penalty=0.3,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        # print(f"generated ids is {generated_ids[0]} has length {len(generated_ids[0])}")
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        print(f"generated text is {generated_text[0]} has length {len(generated_text[0])}")
        answer_dict = {"number": number, "title": title, "answer": generated_text[0],
                           "test": test_list}
        answer_dict_list.append(answer_dict)
        counter += 1
        model.clear_early_exit_layer_indices()

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

    end = time.time()
    print(f"Time to generate HumanEval for 16B with w5_e18 is {end - start} seconds")

if __name__== "__main__":
    main()
