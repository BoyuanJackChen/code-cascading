from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os

pe_jv = "<filename>solutions/solution_1.java\n// Here is the correct implementation of the code exercise in java:\n"


def main():
    number_key = "task_id"
    prompt_key = "prompt"

    output_file_name = 'starcoder_java_raw.json'
    existing_data = []
    existing_numbers = []
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r') as f:
            existing_data = json.load(f)
        for data_dict in existing_data:
            existing_numbers.append(data_dict["number"])
        

    # Initialize model and tokenizer
    checkpoint = "bigcode/starcoder"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    model.eval()
    print(f"Time to load model is {time.time() - start_load_model}")

    # Load Leetcode Dataset
    all_questions_dict = json.load(open("../data/HumanEval_java.json", "r"))

    # Generate answers
    answer_dict_list = []
    counter = 0

    start = time.time()
    for question in all_questions_dict:
        number = question[number_key]
        if number in existing_numbers:
            print(f"Skipping question {number}")
            continue
        prompt = question[prompt_key]
        prompt = prompt.replace("\t", "    ")
        prompt += pe_jv
        print(f"On question {number}")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"input ids is {input_ids} has length {len(input_ids)}")
        generated_ids = model.generate(
            input_ids,
            max_length=1000,
            num_beams=4,
            num_beam_groups=4,
            diversity_penalty=0.3,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        # print(f"generated text is {generated_text[0]} has length {len(generated_text[0])}")
        answer_dict = {
            "number": number, 
            "prompt": prompt,
            "answer": generated_text[0],
            "text": question["text"],
            "test": question["test"],
            "canonical_solution": question["canonical_solution"],
            "declaration": question["declaration"],
            "example_test": question["example_test"],
        }
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

    end = time.time()
    print(f"Time to generate HumanEval-java for starcoder with full capacity is {end - start} seconds")

if __name__== "__main__":
    main()
