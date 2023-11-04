from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import multiprocessing

stopwords = ["\n\n\n", "\n\n"]
imports = "import math\nfrom typing import List\n"
checking_end = "pass\n\n# Assume the above function is completed. Write 3 lines of testing code for the function.\n\nassert"
feedback_unspecific = "\n\n# Above is the answer you just generated, but it was actually incorrect"
can_you_fix = "Can you fix it?\n"

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        # Delete "<|endoftext|>" in answer, if it is in it
        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
        answer = answer[:min_i]
        result.append(answer)
    return result


def process_error_message(error_message, testcode):
    if error_message.startswith("Error occurred: "):
        error_message = error_message[len("Error occurred: "):]
    if testcode.startswith("\n"):
        testcode = testcode[1:]
    line_idx = -1
    num_map = {
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5
    }
    for key in num_map:
        if key in error_message:
            line_idx = num_map[key]
    if line_idx >= 0:
        testcode_trimmed = testcode.strip().split('\n')
        return f", because it couldn't pass the test {testcode_trimmed[line_idx]}. "
    elif len(error_message) <= 1:
        return ". "
    else:
        return f", because of {error_message}. "


if __name__== "__main__":
    checkpoint = "Salesforce/codegen-16B-mono"
    output_file_name = "16B_specific_output.json"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")
    model.eval()

    # Load the questions
    all_questions_dict = json.load(open("../evaluations/humaneval/data/HumanEval_py.json", "r"))

    answer_dict_list = []
    start_time = time.time()
    for question_dict in all_questions_dict:
        number = question_dict["task_id"]
        number_int = int(number.split('/')[1])
        max_new_tokens = 200
        if number_int == 81:
            max_new_tokens = 300
        print(f"\n\n\n--- On Question {number} ---")
        prompt = question_dict["prompt"]

        # Get the def line
        clean_prompt = ""
        prompt_list = prompt.split("\n")
        for idx, line in enumerate(prompt_list):
            if line.strip().startswith('def'):
                clean_prompt = '\n'.join(prompt_list[idx:])

        prompt_original = prompt + ""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        start_generating = time.time()

        # Greedy search for pass@1 generations
        generated_ids = model.generate(
            input_ids,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        # print(f"generated_text is:\n{generated_text[0]}")
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        generated_len = len(decoded_list) - len(input_ids[0])
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt = tokenizer.decode(prompt_ids[0])
        trimmed_text = trim_with_stopwords(generated_text, stopwords, prompt)[0]
        print(f"First round answer is:\n{trimmed_text}")
        answer_dict = {'number': number, 
                       'prompt': prompt_original, 
                       'first_answer': trimmed_text}

        # Now generate a set of 3 tests
        prompt_testcode = prompt_original + checking_end
        input_ids = tokenizer(prompt_testcode, return_tensors="pt").input_ids.to('cuda')
        testcode_ids = model.generate(
            input_ids,
            use_cache = True,
            pad_token_id = tokenizer.eos_token_id,
            max_new_tokens = 3*32,
            temperature = 0.0,
            num_beams = 1
        )
        testcode_text = tokenizer.batch_decode(testcode_ids, skip_special_tokens=False)[0]
        testcode_trimmed = "\nassert" + testcode_text.split("\nassert", 1)[1]
        testcode_trimmed_list = testcode_trimmed.split("\n")[:4]
        testcode_trimmed = "\n".join(testcode_trimmed_list)

        # Now run the test and see if it passes; then extract the error message
        answer_textcode = imports + prompt + trimmed_text + "\n" + testcode_trimmed
        correct = False
        error_message = ""
        
        def code_to_run(result_queue):
            try:
                exec(answer_textcode, globals())
                result_queue.put(True)
            except Exception as e:
                result_queue.put(False)
        
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
        process.start()
        process.join(3)
        if process.is_alive():
            error_message = "Execution timeout."
            process.terminate()
            correct = False
        else:
            result = result_queue.get()
            if isinstance(result, tuple) and not result[0]:  # This means an error occurred
                correct = False
                error_message = result[1]
            else:
                correct = result
        process.close()

        # If the first round answer is correct we just end it.        
        if correct: 
            answer_dict["second_answer"] = answer_dict["first_answer"]
            print(f"First answer is correct.")
        # If the first round answer is incorrect, we generate a second round answer.
        else:
            # Process error message
            print(f"\nOriginal error message: {error_message}")
            error_message = process_error_message(error_message, testcode_trimmed)
            print(f"New error message: {error_message}\n")

            # Generate the second round of answers with new feedback
            new_text = prompt_original + trimmed_text
            prompt = new_text + feedback_unspecific + error_message + can_you_fix + clean_prompt
            print(f"New prompt is:\n{prompt}")
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            generated_ids = model.generate(
                input_ids,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                num_beams=1
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            decoded_list = []
            for ids in generated_ids[0]:
                word = tokenizer.decode(int(ids))
                decoded_list.append(word)
            generated_len = len(decoded_list) - len(input_ids[0])
            prompt = tokenizer.decode(input_ids[0])
            # print(f"generated new text is:\n{generated_text[0]}")
            trimmed_text = trim_with_stopwords(generated_text, stopwords, prompt)[0]
            print(f"Second round of answer is:\n{generated_text[0]}")
            answer_dict['second_answer'] = trimmed_text

        # Append auxiliary stuff to the answer to this question
        answer_dict['feedback'] = feedback_unspecific
        answer_dict["test"] = question_dict["test"],
        answer_dict["canonical_solution"] = question_dict["canonical_solution"]
        answer_dict["declaration"] = question_dict["declaration"]
        answer_dict_list.append(answer_dict)

        # Update to json file
        if not os.path.exists(output_file_name):
            output_data = [answer_dict]
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
            answer_dict_list = []
        else:
            with open(output_file_name, 'r') as f:
                output_data = json.load(f)
            output_data += answer_dict_list
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
            answer_dict_list = []

    print(f"Time to generate is {time.time() - start_time} seconds")