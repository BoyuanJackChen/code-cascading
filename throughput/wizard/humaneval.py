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
parser.add_argument("--batch_size", type=int, default=3, help="Batch size for number of questions")
parser.add_argument("--start_num", type=int, default=0, help="Model name")
parser.add_argument("--pass_at", type=int, default=10, help="pass @ how many")
parser.add_argument("--test_lines", type=int, default=1, help="How many lines of assert do we want eventually")
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
    all_questions_dict = json.load(open("../../evaluations/humaneval/data/HumanEval_py.json", "r"))

    # Prepare the model checkpoint (just 1)
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
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto", padding_side='left')
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

    batch_size = args.batch_size
    pass_at = args.pass_at
    start_question = args.start_num
    end_question = start_question + batch_size
    # Create a sub dictionary based on batch size
    sub_dict = all_questions_dict[start_question:end_question]
    all_prompts_text = []
    all_tests_text = []
    
    time_file_name = f'time_{model_size}_p{pass_at}_{start_question}-{end_question}.txt'
    output_file_name = f'{model_size}_p{pass_at}_{start_question}-{end_question}.json'
    
    # Create prompt and testcode batches from given questions
    for question in sub_dict:
        prompt = question[prompt_key]
        if pass_at == 0:
            all_prompts_text += [prompt]
        else:
            all_prompts_text += [prompt] * pass_at
            all_tests_text += [prompt + checking_end]
    
    # print(f"Model is {checkpoint}")
    # print(f"Pass@{args.pass_at}")
    # print(f"testlines {args.test_lines}")
    # print(f"start {start_question}")
    # print(f"end {end_question}")
    # print(f"batch size {args.batch_size}")
    # print(f"all_prompts_text has length {len(all_prompts_text)}")
    # print(f"all_tests_text has length {len(all_tests_text)}")
    prompt_ids = tokenizer.batch_encode_plus(all_prompts_text, padding=True, return_tensors="pt").to(torch.cuda.current_device())
    logits_processor = LogitsProcessorList([StopSequences(stop_words_ids, batch_size=len(all_prompts_text), encounters=1)])
    
    generation_start = time.time()
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
    all_answer_trimmed = [trim_substring_from_end(answer, eof_token) for answer in answer_text]
    torch.cuda.empty_cache()
    print(f"Answer generation done")
    
    correct = False
    pass_idx = 0
    if pass_at == 0:
        pass_idx += 1
        this_answer = [" " + answer_trimmed for answer_trimmed in all_answer_trimmed]
        testcode_trimmed = ""
    else:
        prompt_ids = tokenizer.batch_encode_plus(all_tests_text, padding=True, return_tensors="pt").to(torch.cuda.current_device())
        logits_processor = LogitsProcessorList([StopSequences(assert_stop_words_ids, batch_size=len(all_tests_text), encounters=args.test_lines+1)])
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
        all_testcode_text = tokenizer.batch_decode(testcode_ids, skip_special_tokens=False)
        all_testcode_trimmed = []
        for testcode_text in all_testcode_text:
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
            all_testcode_trimmed.append(testcode_trimmed)
        torch.cuda.empty_cache()
        
        def code_to_run(idx, answer_textcode, result_queue):
            try:
                exec(answer_textcode, globals())
                result_queue.put(idx)
            except Exception as e:
                result_queue.put(-1)

        def run_in_parallel(all_answer_trimmed, imports, all_prompts_text, all_testcode_trimmed, pass_at):
            processes = []
            result_queue = multiprocessing.Queue()

            for idx, this_answer in enumerate(all_answer_trimmed):
                test_idx = int(idx / pass_at)
                testcode_trimmed = all_testcode_trimmed[test_idx]
                prompt = all_prompts_text[idx]
                answer_textcode = imports + prompt + this_answer + "\n" + testcode_trimmed
                process = multiprocessing.Process(target=code_to_run, args=(idx, answer_textcode, result_queue))
                processes.append(process)

            for process in processes:
                process.start()

            correct_idx = []
            for process in processes:
                process.join(3)
                if process.is_alive():
                    process.terminate()
                process.join()

                # Collect results
                while not result_queue.empty():
                    result = result_queue.get()
                    print(result)
                    if result >= 0:  # If any of the processes returned True
                        correct_idx.append(result)  # Store the index of the correct answer
                        # break  # Exit the loop as we have got a correct answer.

            # Terminating other processes as we have a correct answer.
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()

            return correct_idx  # Return the index of the correct answer or None if no correct answer was found

        print(f"all_answer_trimmed has length {len(all_answer_trimmed)}")
        print(f"all_prompts_text has length {len(all_prompts_text)}")
        print(f"all_testcode_trimmed has length {len(all_testcode_trimmed)}")
        print(f"pass_at is {pass_at}")
        all_correct_idx = run_in_parallel(all_answer_trimmed, imports, all_prompts_text, all_testcode_trimmed, pass_at)
        all_correct_idx.sort()
        print(all_correct_idx)
        
    print(f"Total time took {time.time() - generation_start} seconds")
            
        # answer_dict = {
        #     "number": number,
        #     "prompt": prompt,
        #     "checkpoint": model_size,
        #     "pass": pass_idx,
        #     "correct": correct,
        #     "answer": this_answer,
        #     "generated_testcode": testcode_trimmed,
        #     "test": question["test"],
        #     "canonical_solution": question["canonical_solution"],
        #     "declaration": question["declaration"]
        # }
        # answer_dict_list.append(answer_dict)
        # counter += 1
        # pass_idx += 1

        # # Write to json file by loading and appending towards the end
        # if not os.path.exists(output_file_name):
        #     output_data = [answer_dict]
        #     with open(output_file_name, 'w') as f:
        #         json.dump(output_data, f, indent=4)
        #     answer_dict_list = []
        # elif counter >= 1:
        #     with open(output_file_name, 'r') as f:
        #         output_data = json.load(f)
        #     output_data += answer_dict_list
        #     with open(output_file_name, 'w') as f:
        #         json.dump(output_data, f, indent=4)
        #     answer_dict_list = []
            
        # # Write the time results to a .txt file
        # total_end = time.time()
        # with open(time_file_name, 'a') as f:
        #     f.write(f"Loop {loop}, total time taken: {total_end - loop_start} seconds\n")

if __name__== "__main__":
    main(FLAGS)
