from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import argparse
import multiprocessing

# Create some argparsers
parser = argparse.ArgumentParser(description='Generate answers for HumanEval')
parser.add_argument('--cascade', type=bool, default=True, help='Should we do cascade')
parser.add_argument('--use_350M', type=int, default=1, help='Should we include 350M model or not')
parser.add_argument('--answer_max', type=int, default=300, help='The max number of tokens for answers')
parser.add_argument('--test_lines', type=int, default=4, help='The max number of tokens for tests')
FLAGS = parser.parse_args()

checking_end = "pass\n\n# Assume the above function is completed. Write 3 lines of testing code for the function.\n\nassert"
stop_words = ["\n\n\n", "\n\n"]
imports = "\nimport math\nfrom typing import List\n"

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
        answer = answer[:min_i]
        result.append(answer)
    return result

def main(args):
    number_key = "task_id"
    prompt_key = "prompt"
    temperature = 0.8
    if args.test_lines > 100:
        temperature = 0.0
    test_lines = args.test_lines % 100
    if args.use_350M:
        output_file_name = f'full_cascade_{args.test_lines}tests.json'
    else:
        output_file_name = f'no_350M_{args.test_lines}tests.json'
    existing_data = []
    existing_numbers = []
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r') as f:
            existing_data = json.load(f)
        for data_dict in existing_data:
            existing_numbers.append(data_dict["number"])

    # Load HumanEval Dataset
    all_questions_dict = json.load(open("../../../evaluations/humaneval/data/HumanEval_py.json", "r"))

    # Generate answers
    answer_dict_list = []
    counter = 0
    
    # Load all the models
    model_1 = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono", device_map="auto")
    model_1.eval()
    model_2 = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-mono", device_map="auto")
    model_2.eval()
    model_3 = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-6B-mono", device_map="auto")
    model_3.eval()
    model_4 = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-16B-mono", device_map="auto")
    model_4.eval()
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-16B-mono", device_map="auto")

    # Generate answers
    total_start = time.time()
    for question in all_questions_dict:
        all_checkpoints = ["16B", "6B", "2B"]
        if args.use_350M:
            all_checkpoints.append("350M")
        number = question[number_key]
        number_int = int(number.split('/')[1])
        if number in existing_numbers:
            print(f"Skipping question {number}")
            continue
        print(f"On question {number}")
        undone = True
        max_new_tokens = 200
        if number_int == 81:
            max_new_tokens = 300
        
        while undone:
            prompt = question[prompt_key]
            # Initialize model and tokenizer
            model_size = all_checkpoints.pop()
            if model_size == "350M":
                model = model_1
            elif model_size == "2B":
                model = model_2
            elif model_size == "6B":
                model = model_3
            elif model_size == "16B":
                model = model_4
            
            # Generate solutions
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
            answer_ids = model.generate(
                input_ids,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens = max_new_tokens,
                top_k = 0,
                top_p = 0.95,
                temperature = temperature,
                num_beams = 1
            )
            answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
            prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
            answer_trimmed = trim_with_stopwords(answer_text, stop_words, prompt)[0]
            
            # Generate test code
            prompt_testcode = prompt + checking_end
            input_ids = tokenizer(prompt_testcode, return_tensors="pt").input_ids.to('cuda')
            testcode_ids = model.generate(
                input_ids,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=test_lines * 32,
                top_k = 0,
                top_p = 0.95,
                temperature = temperature,
                num_beams=1,
            )
            testcode_text = tokenizer.batch_decode(testcode_ids, skip_special_tokens=False)[0]
            testcode_trimmed = "\nassert" + testcode_text.split("\nassert", 1)[1]
            testcode_trimmed_list = testcode_trimmed.split("\n")[:test_lines+1]
            testcode_trimmed = "\n".join(testcode_trimmed_list)
            print(f"HumanEval question {number_int}, Codegen {model_size}:")
            answer_textcode = imports + prompt + answer_trimmed + "\n" + testcode_trimmed
            print(answer_textcode)
            
            def code_to_run(result_queue):
                try:
                    exec(answer_textcode, globals())
                    print("No error occurred!\n")
                    result_queue.put(True)
                except Exception as e:
                    print(f"Error occurred: {e}\n")
                    result_queue.put(False)

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
            process.start()
            process.join(3)
            if process.is_alive():
                print("Code took too long to run!")
                process.terminate()
                correct = False
            else:
                correct = result_queue.get()
                if correct:
                    undone = False
            process.close()

            answer_dict = {
                "number": number,
                "prompt": prompt,
                "checkpoint": model_size,
                "passed": (not undone),
                "answer": answer_trimmed,
                "generated_testcode": testcode_trimmed,
                "test": question["test"],
                "canonical_solution": question["canonical_solution"],
                "declaration": question["declaration"],
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
            
            if len(all_checkpoints) == 0:
                undone = False
    total_end = time.time()
    print(f"\nTotal time taken: {total_end - total_start} seconds")
    print(f"Max answer tokens: {args.answer_max}")
    print(f"Max test lines: {test_lines}")
    print(f"Max test tokens: {test_lines * 32}")
    print(f"temperature: {temperature}")
        

if __name__== "__main__":
    main(FLAGS)
