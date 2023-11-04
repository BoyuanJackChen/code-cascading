from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
from langdetect import detect
import time
import json
import os
import argparse
import multiprocessing
import torch
import re

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=0, help="Model name")
parser.add_argument("--pass_at", type=int, default=-1, help="pass @ how many")
parser.add_argument("--test_lines", type=int, default=1, help="Number of lines of test code to generate")
parser.add_argument("--num_loops", type=int, default=10, help="Number of times that we do this experiment")
FLAGS = parser.parse_args()

stop_words = ["\n\n", ("\n","\n")]
assert_stop_words = ["assert"] + stop_words
eof_id = 50256
eof_token = "<|endoftext|>"
general_def_line = "\n\n# Start your code here\ndef function(input_string):\n    "

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    return answer

def is_english(text):
    try:
        language = detect(text)
        return language == 'en'
    except:
        return False
    
def trim_starcoder_assert(answer):
    pattern = r',\s*(\"[^\"\\]*(?:\\.[^\"\\]*)*\")?\s*(\r?\n)?\s*$'
    new_answer = re.sub(pattern, '', answer)
    if ".format" in new_answer:
        new_answer = new_answer[:new_answer.find(", \"")]
    return new_answer

def contains_input_assignment(canonical_solution):
    canonical_solution = canonical_solution.replace("\\n", "\n")
    canonical_solution_lines = canonical_solution.split('\n')
    for line in canonical_solution_lines:
        if "=" in line and "input()" in line:
            return True
    return False

def main(args):
    loading_start = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    checking_end = f"pass\n\n# Assume the above function is completed. Write {args.test_lines} line of testing code for the function.\n\nassert "

    # Load APPS dataset
    all_questions_dict = load_dataset("codeparrot/apps", split="test")
    number_key = "problem_id"
    prompt_key = "question"
    difficulty_key = "difficulty"
    solutions_key = "solutions"
    test_key = "input_output"
    startercode_key = "starter_code"
    
    # Prepare the model checkpoint (just 1)
    answer_dict_list = []
    counter = 0
    if args.model == 0:
        model_size = "350M"
    elif args.model == 1:
        model_size = "2B"
    elif args.model == 2:
        model_size = "6B"
    elif args.model == 3:
        model_size = "16B"
    checkpoint = f"Salesforce/codegen-{model_size}-mono"

    # Make directory if f"{model_size}" dir does not exist
    if not os.path.exists(f"{model_size}"):
        os.mkdir(f"{model_size}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    loading_end = time.time()
    print(f"Time to load model is {loading_end - loading_start}")
    
    # Stopping criteria for generation using the LogitsProcessor class
    class StopSequences(LogitsProcessor):
        def __init__(self, stop_sequences, batch_size, encounters=1, eos_token_id=50256):
            StoppingCriteria.__init__(self)
            self.stop_sequences = tokenizer.batch_encode_plus(stop_sequences)['input_ids']
            self.batch_size = batch_size
            self.encounters = [encounters] * batch_size
            self.NUM_ENCOUNTERS = encounters
            self.eos_token_id = eos_token_id
            self.just_started = True

        def __call__(self, input_ids, scores):
            if self.just_started:
                self.just_started = False
                return scores
            forced_eos = torch.full((scores.size(1),), -float("inf"))
            forced_eos[self.eos_token_id] = 0
            for stop in self.stop_sequences:
                # Check if the input_ids end with the stop sequence
                for i in range(self.batch_size):
                    if self.encounters[i] <= 0:
                        continue
                    if input_ids[i][-len(stop):].tolist() == stop:
                        self.encounters[i] -= 1
                        if self.encounters[i] <= 0:
                            scores[i] = forced_eos
            return scores

    all_ks = [args.pass_at] if args.pass_at >= 0 else [0,1,2,3,4,5,10]
    for pass_at in all_ks:
        num_loops = 1 if pass_at in [0,1] else args.num_loops
        # Since it is sampling with temperature, do it for multiple loops to find average
        for loop in range(num_loops):
            time_file_name = f'time_{model_size}_p{pass_at}_t{args.test_lines}.txt'
            output_file_name = f'{model_size}/{model_size}_p{pass_at}_t{args.test_lines}_l{loop}.json'
            max_seen_number = -1
            if os.path.exists(output_file_name):
                if os.path.exists(f'{model_size}/{model_size}_p{pass_at}_t{args.test_lines}_l{loop+1}.json'):
                    continue
                else:
                    last_generated_data = json.load(open(output_file_name, "r"))
                    for answer_dict in last_generated_data:
                        if answer_dict["number"] > max_seen_number:
                            max_seen_number = answer_dict["number"]
            loop_start = time.time()
            
            # Go through each question
            for question in all_questions_dict:
                number = question[number_key]
                if number < 4000:
                    continue
                if int(number) <= max_seen_number:
                    continue
                prompt = question[prompt_key]
                if not is_english(prompt):
                    continue
                original_prompt = prompt + ""
                
                # Find the shortest canonical answer (APPS-only)
                canonical_idx = 0
                shortest_len = 100000
                canonical_collage = question[solutions_key][2:-2]
                canonical_solution_list = canonical_collage.split('", "')
                for idx in range(len(canonical_solution_list)):
                    if len(canonical_solution_list[idx]) < shortest_len and contains_input_assignment(canonical_solution_list[idx]):
                        shortest_len = len(canonical_solution_list[idx])
                        canonical_idx = idx
                canonical_solution = canonical_solution_list[canonical_idx]
                canonical_solution = canonical_solution.replace("\\n", "\n")
                
                # Get the input processing line. Replace that with a variable (APPS-only)
                input_line = ""
                canonical_solution_lines = canonical_solution.split('\n')
                for line in canonical_solution_lines:
                    if "=" in line and "input()" in line:
                        input_line = line
                        break
                input_line = input_line.replace("input()", "input_string")
                while input_line.startswith(" ") or input_line.startswith("\t"):
                    input_line = input_line[1:]
                
                # Decide def_line
                if question["starter_code"] == "":
                    def_line = general_def_line
                    def_name = "function("
                else:
                    lines = question["starter_code"].split('\n')
                    for line in reversed(lines):
                        if "def " in line:
                            def_line = line.lstrip()
                            def_name = def_line[def_line.find("def")+4:def_line.find("(")] + "("
                            break
                
                # Process canonical tests to canonical inputs and outputs (APPS-only)
                prompt += def_line + input_line + "\n    "
                prompt_ids = tokenizer.batch_encode_plus([prompt]*max(pass_at,1), return_tensors="pt").to(torch.cuda.current_device())
                if len(prompt_ids[0]) > 2048:
                    # Remove everything from "-----Examples-----"
                    original_prompt = original_prompt.split("-----Examples-----")[0]
                    prompt = original_prompt + def_line + input_line + "\n    "
                logits_processor = LogitsProcessorList([StopSequences(stop_words, batch_size=max(pass_at,1), encounters=1)])
                
                # Generate answers
                max_new_tokens = min(2048-len(prompt_ids[0]), 512)
                print(f"On question {number}, max_new_tokens {max_new_tokens}")
                with torch.no_grad():
                    if pass_at in [0,1]:
                        answer_ids = model.generate(
                            **prompt_ids,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = max_new_tokens,
                            num_beams = 1,
                            do_sample = False,
                            logits_processor = logits_processor
                        )
                    else:
                        answer_ids = model.generate(
                            **prompt_ids,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = max_new_tokens,
                            do_sample = True,
                            top_k = 0,
                            top_p = 0.95,
                            temperature = 0.8,
                            num_beams = 1,
                            logits_processor = logits_processor
                        )
                answer_ids = answer_ids[:, len(prompt_ids['input_ids'][0]):]
                answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
                answer_trimmed = [trim_substring_from_end(answer, eof_token) for answer in answer_text]                
                torch.cuda.empty_cache()
                
                correct = False
                pass_idx = 0
                if pass_at == 0:
                    pass_idx += 1
                    this_answer = answer_trimmed[0]
                    testcode_trimmed = ""
                else:
                    # Generate test code with greedy search. Here we just generate one line, and we pick 
                    # the one answer that passes it; otherwise we just use the last one.
                    prompt_testcode = original_prompt + def_line + checking_end + def_name + '\"'
                    print(f"test prompt for question {number} is:\n{prompt_testcode}")
                    prompt_ids = tokenizer.batch_encode_plus([prompt_testcode], return_tensors="pt").to(torch.cuda.current_device())
                    logits_processor = LogitsProcessorList([StopSequences(assert_stop_words, batch_size=1, encounters=1)])
                    max_new_tokens = min(2048-len(prompt_ids[0]), 100*args.test_lines)
                    with torch.no_grad():
                        testcode_ids = model.generate(
                            **prompt_ids,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = max_new_tokens,
                            do_sample = False,
                            num_beams = 1,
                            logits_processor = logits_processor
                        )
                    testcode_ids = testcode_ids[:, len(prompt_ids[0]):]
                    testcode_text = tokenizer.batch_decode(testcode_ids, skip_special_tokens=False)[0]
                    print(f"testcode_text is \n{testcode_text}")
                    if testcode_text.endswith(eof_token) or testcode_text.endswith("assert"):
                        testcode_text = trim_substring_from_end(testcode_text, eof_token)
                        testcode_text = trim_substring_from_end(testcode_text, "assert")
                        if testcode_text.startswith(def_name[:-1]):
                            testcode_trimmed = "assert " + def_name + "\"" + testcode_text[len(def_name):]
                            testcode_trimmed = testcode_trimmed[:testcode_trimmed.rfind(")")]
                        else:
                            testcode_trimmed = "assert " + def_name + "\"" + testcode_text
                    else: 
                        # In this case the model generated insufficient testlines and started generating irrelevant contents. We trim off at the last line with assert
                        testcode_text = testcode_text.split('\n')[0]
                        if testcode_text.startswith(def_name[:-1]):
                            testcode_trimmed = "assert " + def_name + "\"" + testcode_text[len(def_name):]
                            testcode_trimmed = testcode_trimmed[:testcode_trimmed.rfind(")")]
                        else:
                            testcode_trimmed = "assert " + def_name + "\"" + testcode_text
                        testcode_trimmed = testcode_trimmed[:testcode_trimmed.find("#")]
                        testcode_trimmed = trim_starcoder_assert(testcode_trimmed)
                    torch.cuda.empty_cache()
                    
                    if testcode_trimmed == "":
                        correct = False
                        this_answer = answer_trimmed[-1]
                        pass_idx = len(answer_trimmed)
                    else:
                        parts = testcode_trimmed.split("assert")
                        testcode_trimmed = "assert" + parts[1]
                        for this_answer in answer_trimmed:
                            pass_idx += 1
                            answer_textcode = def_line[def_line.find("def"):] + input_line + "\n    " + this_answer + "\n" + testcode_trimmed
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
                                # print("Code took too long to run!")
                                process.terminate()
                                process.join()
                                correct = False
                            else:
                                correct = result_queue.get()
                            process.close()
                            
                            if correct:
                                break
                
                answer_dict = {
                    "number": number,
                    # "prompt": prompt,
                    "checkpoint": model_size,
                    "pass": pass_idx,
                    "correct": correct,
                    "answer": def_line[def_line.find("def"):] + input_line + "\n    " + this_answer,
                    "generated_testcode": testcode_trimmed,
                    # "canonical_solution": canonical_solution,
                    # "canonical_inputs": canonical_inputs,
                    # "canonical_outputs": canonical_outputs,
                    "difficulty": question[difficulty_key]
                }
                answer_dict_list.append(answer_dict)
                counter += 1
                pass_idx += 1

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
                
                # Free up VRAM
                torch.cuda.empty_cache()
                
            total_end = time.time()
            print(f"\nLoop {loop}, total time taken: {total_end - loop_start} seconds")
            print(f"This is the result of {model_size}, pick@{pass_at}, testlines {args.test_lines}")
            
            # Write the time results to a .txt file
            with open(time_file_name, 'a') as f:
                f.write(f"Loop {loop}, total time taken: {round(total_end - loop_start, 1)} seconds, "
                        f"which is {round((total_end - loop_start)/3600, 1)} hours\n")
        

if __name__== "__main__":
    main(FLAGS)
