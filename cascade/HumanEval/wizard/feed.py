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
parser.add_argument("--model", type=int, default=3, help="Model name")
parser.add_argument("--pass_at", type=int, default=1, help="pass @ how many")
parser.add_argument("--test_lines", type=int, default=1, help="Number of lines of test code to generate")
parser.add_argument("--num_loops", type=int, default=10, help="Number of times that we do this experiment")
FLAGS = parser.parse_args()

# We will hard-code the stop tokens for llama code family, as the tokenizer is automatically adding start tokens
stop_words = ["\n\n", ("\n","\n")]
stop_words_ids = [[13,13]]
assert_stop_words = ["assert"] + stop_words
assert_stop_words_ids = [[9294]] + stop_words_ids
eof_id = 2
eof_token = "</s>"
imports = "\nimport math\nfrom typing import List\n"

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    if answer.startswith("   ") and not answer.startswith("    "):
        answer = " " + answer
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
        model_size = "1B"
        checkpoint = "WizardLM/WizardCoder-1B-V1.0"
    elif args.model == 1:
        model_size = "3B"
        checkpoint = "WizardLM/WizardCoder-3B-V1.0"
    elif args.model == 2:
        model_size = "7B"
        checkpoint = f"WizardLM/WizardCoder-Python-7B-V1.0"
    elif args.model == 3:
        model_size = "13B"
        checkpoint = f"WizardLM/WizardCoder-Python-13B-V1.0"
    elif args.model == 4:
        model_size = "15B"
        checkpoint = "WizardLM/WizardCoder-15B-V1.0"
    elif args.model == 5:
        model_size = "34B"
        checkpoint = f"WizardLM/WizardCoder-Python-34B-V1.0"
    print(f"Model is {checkpoint}")
    print(f"Pass @ {args.pass_at}")
    print(f"testlines {args.test_lines}")
    print(f"num loops {args.num_loops}")

    # Make directory if f"{model_size}" dir does not exist
    if not os.path.exists(f"{model_size}_feed"):
        os.mkdir(f"{model_size}_feed")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    loading_end = time.time()
    print(f"Time to load model is {loading_end - loading_start}")
    
    # Stopping criteria for generation using the LogitsProcessor class
    class StopSequences(LogitsProcessor):
        def __init__(self, stop_ids, batch_size, encounters=1, eos_token_id=2):
            StoppingCriteria.__init__(self)
            self.stop_sequences = stop_ids
            self.batch_size = batch_size
            self.encounters = [encounters] * batch_size
            self.NUM_ENCOUNTERS = encounters
            self.eos_token_id = eos_token_id

        def __call__(self, input_ids, scores):
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

    all_k_list = [args.pass_at] if args.pass_at >= 0 else [0,1,2,3,4,5,10]
    for pass_at in all_k_list:
        num_loops = 1 if pass_at in [0,1] else args.num_loops
        # Since it is sampling with temperature, do it for multiple loops to find average
        for loop in range(num_loops):
            # time_file_name = f'time_{model_size}_p{pass_at}_t{args.test_lines}.txt'
            output_file_name = f'{model_size}_feed/{model_size}_p{pass_at}_t{args.test_lines}_l{loop}.json'
            if os.path.exists(output_file_name):
                continue
            loop_start = time.time()
            all_prev_answers = json.load(open(f"./wave_1_jubail_sep/7B/7B_p5_t1_l{loop}.json", "r"))
            
            # Go through each question
            for prev_question in all_prev_answers:
                number = prev_question["number"]
                number_int = int(number.split("/")[1])
                if prev_question["correct"]:
                    continue
                print(f"On question {number}")
                question = all_questions_dict[number_int]
                prompt = question[prompt_key]
                prompt_lines = prompt.split('\n')
                def_line = ""
                for line in prompt_lines:
                    if line.startswith("def"):
                        def_line = line
                        break
                prev_answer = prev_question["answer"]
                prev_test = prev_question["generated_testcode"]
                feed_prompt = prompt + prev_answer + "\n# The above answer is actually incorrect, because it cannot pass this test: " + prev_test + ". Please fix the answer.\n" + def_line + '\n    """\n    Below is the correct implementation\n    """\n'
                prompt_ids = tokenizer.batch_encode_plus([feed_prompt]*max(pass_at,1), return_tensors="pt").to(torch.cuda.current_device())
                logits_processor = LogitsProcessorList([StopSequences(stop_words_ids, batch_size=pass_at, encounters=1)])
                
                # Generate answers
                with torch.no_grad():
                    if pass_at in [0,1]:
                        answer_ids = model.generate(
                            **prompt_ids,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = 310,
                            num_beams = 1,
                            do_sample = False,
                            logits_processor = logits_processor
                        )
                    else:
                        answer_ids = model.generate(
                            **prompt_ids,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = 310,
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
                    this_answer = " " + answer_trimmed[0]
                    testcode_trimmed = ""
                else:
                    # Generate test code with greedy search. Here we just generate one line, and we pick 
                    # the one answer that passes it; otherwise we just use the last one.
                    prompt_testcode = prompt + checking_end
                    prompt_ids = tokenizer.batch_encode_plus([prompt_testcode], return_tensors="pt").to(torch.cuda.current_device())
                    logits_processor = LogitsProcessorList([StopSequences(assert_stop_words_ids, batch_size=1, encounters=args.test_lines+1)])
                    with torch.no_grad():
                        testcode_ids = model.generate(
                            **prompt_ids,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = 10+50*args.test_lines,
                            do_sample = False,
                            num_beams = 1,
                            logits_processor = logits_processor
                        )
                    testcode_ids = testcode_ids[:, len(prompt_ids[0]):]
                    testcode_text = tokenizer.batch_decode(testcode_ids, skip_special_tokens=False)[0]
                    # Very strange beginning tokenization problem. The first indentation will be translated to 3 spaces
                    if testcode_text.endswith(eof_token) or testcode_text.endswith("assert"):
                        testcode_text = trim_substring_from_end(testcode_text, eof_token)
                        testcode_trimmed = "assert " + trim_substring_from_end(testcode_text, "assert")
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
                    torch.cuda.empty_cache()
                    
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
                    "feed_prompt": feed_prompt,
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

                # Write to json file by loading and appending towards the end
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
                
            # Write the time results to a .txt file
            total_end = time.time()
            # with open(time_file_name, 'a') as f:
            #     f.write(f"Loop {loop}, total time taken: {total_end - loop_start} seconds\n")

if __name__== "__main__":
    main(FLAGS)
