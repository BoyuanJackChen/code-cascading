from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
import time
import json
import os
import argparse
import multiprocessing
import torch

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
imports = "\nimport math\nfrom typing import List\n"

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    return answer

def main(args):
    loading_start = time.time()
    number_key = "task_id"
    prompt_key = "prompt"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    checking_end = f"    pass\n\n# Assume the above function is completed. Write {args.test_lines} lines of testing code for the function.\n\nassert"

    # Load HumanEval Dataset
    all_questions_dict = json.load(open("../../../evaluations/humaneval/data/HumanEval_py.json", "r"))

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
    print(f"Running for {model_size}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left', device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
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

    all_ks = [0,1,2,3,4,5,10]
    if args.pass_at > 0:
        all_ks = [args.pass_at]
    for pass_at in all_ks:
        # Since it is sampling with temperature, do it for multiple loops to find average
        for loop in range(args.num_loops):
            time_file_name = f'batch_{model_size}_p{pass_at}_t{args.test_lines}.txt'
            output_file_name = f'{model_size}_p{pass_at}_t{args.test_lines}_l{loop}.json'
            loop_start = time.time()
            
            # Go through each question
            for question in all_questions_dict:
                number = question[number_key]
                print(f"On question {number}")
                prompt = question[prompt_key]
                prompt_testcode = prompt + checking_end
                if pass_at==0:
                    final_array = [prompt]
                else:
                    final_array = [prompt]*pass_at+[prompt_testcode]
                input_ids = tokenizer.batch_encode_plus(final_array, padding=True, return_tensors="pt").to(torch.cuda.current_device())
                logits_processor = LogitsProcessorList([StopSequences(assert_stop_words, batch_size=pass_at+1, encounters=1)])
                
                # Generate answers
                with torch.no_grad():
                    if pass_at in [0,1]:
                        output_ids = model.generate(
                            **input_ids,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = 300,
                            num_beams = 1,
                            do_sample = False,
                            # num_return_sequences = pass_at,
                            logits_processor = logits_processor
                        )
                    else:
                        output_ids = model.generate(
                            **input_ids,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = 300,
                            do_sample = True,
                            top_k = 0,
                            top_p = 0.95,
                            temperature = 0.8,
                            num_beams = 1,
                            # num_return_sequences = pass_at,
                            logits_processor = logits_processor
                        )
                correct = False
                pass_idx = 0
                if pass_at == 0:
                    pass_idx += 1
                    answer_ids = output_ids[:, len(input_ids['input_ids'][0]):]
                    answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
                    this_answer = [trim_substring_from_end(answer, eof_token) for answer in answer_text][0]
                    testcode_trimmed = ""
                else:
                    testcode_ids = output_ids[-1, len(input_ids['input_ids'][-1]):]
                    answer_ids = output_ids[:-1, len(input_ids['input_ids'][0]):]
                    answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
                    answer_trimmed = [trim_substring_from_end(answer, eof_token) for answer in answer_text]                
                    testcode_text = tokenizer.decode(testcode_ids, skip_special_tokens=False)
                    if testcode_text.endswith(eof_token) or testcode_text.endswith("assert"):
                        testcode_text = trim_substring_from_end(testcode_text, eof_token)
                        testcode_trimmed = "assert" + trim_substring_from_end(testcode_text, "assert")
                    else: 
                        # In this case the model generated insufficient testlines and started generating irrelevant contents. We trim off at the last line with assert
                        testcode_lines = testcode_text.split('\n')
                        assert_lines = []
                        for line in testcode_lines:
                            if line.startswith('assert'):
                                assert_lines.append(line)
                            else:
                                break
                        testcode_trimmed = '\n'.join(assert_lines)
                    
                    for this_answer in answer_trimmed:
                        pass_idx += 1
                        answer_textcode = imports + prompt + this_answer + "\n" + testcode_trimmed
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
                    "prompt": prompt,
                    "checkpoint": model_size,
                    "pass": pass_idx,
                    "correct": correct,
                    "answer": this_answer,
                    "generated_testcode": testcode_trimmed,
                    "test": question["test"],
                    "canonical_solution": question["canonical_solution"],
                    "declaration": question["declaration"]
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
                f.write(f"Loop {loop}, total time taken: {total_end - loop_start} seconds\n")
        

if __name__== "__main__":
    main(FLAGS)
