from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import time
import json
import os
import torch

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def main():
    number_key = "task_id"
    prompt_key = "prompt"

    output_file_name = 'py_top_p.json'
    existing_data = []
    existing_numbers = []
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r') as f:
            existing_data = json.load(f)
        for data_dict in existing_data:
            existing_numbers.append(data_dict["number"])
        
    # Initialize model and tokenizer
    checkpoint = "Salesforce/codegen-16B-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    model.eval()
    print(f"Time to load model is {time.time() - start_load_model}")
    
    stop_words = ["\n\n", "def"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]

    # Load Leetcode Dataset
    all_questions_dict = json.load(open("../data/HumanEval_py.json", "r"))

    # Generate answers
    answer_dict_list = []
    counter = 0

    total_time = 0.0
    for question in all_questions_dict:
        number = question[number_key]
        number_int = int(number.split('/')[1])
        if number in existing_numbers:
            print(f"Skipping question {number}")
            continue
        # Prepare for prompt
        prompt = question[prompt_key]
        print(f"On question {number}; so far took {total_time} seconds")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        # Stopping Criteria
        encounters = 0
        for stop_word_id in stop_words_ids:
            stop_count = (stop_word_id == input_ids[0]).sum().item()
            encounters += stop_count
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=encounters)])

        start = time.time()
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=500,
            temperature = 0.2,
            top_k = 0,
            top_p = 0.95,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
        question_time = time.time() - start
        total_time += question_time
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
        answer_dict = {
            "number": number, 
            "prompt": prompt,
            "answer": generated_text[0],
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
            answer_dict_list = []
        elif counter >= 1:
            with open(output_file_name, 'r') as f:
                output_data = json.load(f)
            output_data += answer_dict_list
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
            answer_dict_list = []

    end = time.time()
    print(f"- Timing 32float - Time to generate HumanEval-python for starcoder with full capacity is {end - start} seconds")

if __name__== "__main__":
    main()
