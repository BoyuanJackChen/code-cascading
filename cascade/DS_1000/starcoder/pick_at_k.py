import tensorflow as tf
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
parser.add_argument("--model", type=int, default=2, help="Model name")
parser.add_argument("--pass_at", type=int, default=-1, help="pass @ how many")
parser.add_argument("--test_lines", type=int, default=1, help="Number of lines of test code to generate")
parser.add_argument("--num_loops", type=int, default=5, help="Number of times that we do this experiment")
FLAGS = parser.parse_args()

# We will hard-code the stop tokens for llama code family, as the tokenizer is automatically adding start tokens
stop_words = ["\n\n", ("\n","\n"), "</code>", "END SOLUTION", "# END SOLUTION"]
stop_words_ids = [[13,13], [829, 401, 29958], [11056, 317, 5607, 2692, 2725], [11794, 317, 5607, 2692, 2725]]
assert_stop_words = ["assert"] + stop_words
assert_stop_words_ids = [[9294]] + stop_words_ids
eos_token_id = 0
eos_token = "<|endoftext|>"
imports = "\nimport math\nfrom typing import List\n"
checking_end = "\n# Assume that the above code is completed. Please write a test for the question.\nassert "
pe_py = "\n<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise in python:\n"


# DS-1000
all_libraries = ["Numpy", "Pandas", "Scipy", "Tensorflow"]
all_num_questions = [219, 290, 105, 44]
dataset_dir = "../../../evaluations/ds_1000/ds1000_data"

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    if answer.startswith("   ") and not answer.startswith("    "):
        answer = " " + answer
    return answer

def main(args):
    loading_start = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tf.config.set_visible_devices([], 'GPU')
    
    # Prepare the model checkpoint (just 1)
    answer_dict_list = []
    counter = 0
    if args.model == 0:
        model_size = "1B"
    elif args.model == 1:
        model_size = "3B"
    elif args.model == 2:
        model_size = "15B"
    checkpoint = f"WizardLM/WizardCoder-{model_size}-V1.0"
    print(f"Model is {checkpoint}")
    print(f"Pass @ {args.pass_at}")
    print(f"testlines {args.test_lines}")
    print(f"num loops {args.num_loops}")

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
        def __init__(self, stop_sequences, batch_size, encounters=1, eos_token_id=eos_token_id):
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

    all_k_list = [args.pass_at] if args.pass_at >= 0 else [0,1,2,3,4,5,10]
    for pass_at in all_k_list:
        num_loops = 1 if pass_at in [0,1] else args.num_loops
        # Since it is sampling with temperature, do it for multiple loops to find average
        for loop in range(num_loops):
            output_file_name = f'{model_size}/{model_size}_p{pass_at}_t{args.test_lines}_l{loop}.json'
            last_library = ""
            last_library_index = -1
            last_number = -1
            if os.path.exists(output_file_name):
                if os.path.exists(f'{model_size}/{model_size}_p{pass_at}_t{args.test_lines}_l{loop+1}.json'):
                    continue
                else:
                    last_generated_data = json.load(open(output_file_name, "r"))
                    last_library = last_generated_data[-1]["library"]
                    if last_library in all_libraries:
                        last_library_index = all_libraries.index(last_library)
                    last_number = last_generated_data[-1]["number"]
            
            # Go through each library
            for l in range(len(all_libraries)):
                if l < last_library_index:
                    continue
                library = all_libraries[l]
                full_library_dir = f"{dataset_dir}/{library}/Completion"
                starting_num = last_number+1 if l == last_library_index else 0
                # Go through each question
                for number in range(starting_num, all_num_questions[l]):
                    question_dir = full_library_dir + f"/q{number}/"
                    print(f"On {library} question {number}")
                    with open(question_dir + "prompt.txt", 'r') as f:
                        prompt = f.read()
                    if "<code>" in prompt and "</code>" in prompt:
                        code_body = prompt[prompt.find("<code>")+len("<code>"):prompt.find("</code>")]
                    else:
                        code_body = prompt
                    # prompt += pe_py
                    print(prompt)
                    prompt_ids = tokenizer.batch_encode_plus([prompt]*max(pass_at,1), return_tensors="pt")
                    prompt_ids = prompt_ids.to(torch.cuda.current_device())
                    logits_processor = LogitsProcessorList([StopSequences(stop_words, batch_size=max(pass_at,1), encounters=1)])
                    
                    # Generate answers
                    with torch.no_grad():
                        if pass_at in [0,1]:
                            answer_ids = model.generate(
                                **prompt_ids,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                max_new_tokens = 300,
                                num_beams = 1,
                                do_sample = False,
                                logits_processor = logits_processor
                            )
                        else:
                            answer_ids = model.generate(
                                **prompt_ids,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                max_new_tokens = 300,
                                do_sample = True,
                                top_k = 0,
                                top_p = 0.95,
                                temperature = 0.8,
                                num_beams = 1,
                                logits_processor = logits_processor
                            )
                    answer_ids = answer_ids[:, len(prompt_ids['input_ids'][0]):]
                    answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
                    answer_trimmed = [trim_substring_from_end(answer, eos_token) for answer in answer_text]
                    answer_trimmed = [trim_substring_from_end(answer, "</code>") for answer in answer_trimmed]
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
                        prompt_ids = tokenizer.batch_encode_plus([prompt_testcode], return_tensors="pt")
                        prompt_ids = prompt_ids.to(torch.cuda.current_device())
                        logits_processor = LogitsProcessorList([StopSequences(assert_stop_words, batch_size=1, encounters=1)])
                        with torch.no_grad():
                            testcode_ids = model.generate(
                                **prompt_ids,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                max_new_tokens = 300*args.test_lines,
                                do_sample = False,
                                num_beams = 1,
                                logits_processor = logits_processor
                            )
                        testcode_ids = testcode_ids[:, len(prompt_ids[0]):]
                        testcode_text = tokenizer.batch_decode(testcode_ids, skip_special_tokens=False)[0]
                        # Very strange beginning tokenization problem. The first indentation will be translated to 3 spaces
                        testcode_text = trim_substring_from_end(testcode_text, eos_token)
                        testcode_text = trim_substring_from_end(testcode_text, "assert")
                        testcode_text = trim_substring_from_end(testcode_text, "</code>")
                        testcode_trimmed = "assert " + testcode_text
                        torch.cuda.empty_cache()
                        
                        for this_answer in answer_trimmed:
                            pass_idx += 1
                            answer_textcode = code_body + "\n# answer:\n" + this_answer + "\n\n" + "# test:\n" + testcode_trimmed
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
                        torch.cuda.empty_cache()
                        
                    answer_dict = {
                        "library": library,
                        "number": number,
                        "prompt": prompt,
                        "checkpoint": model_size,
                        "pass": pass_idx,
                        "correct": correct,
                        "answer": this_answer,
                        "generated_testcode": testcode_trimmed
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

if __name__== "__main__":
    main(FLAGS)
