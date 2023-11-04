from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
import time
import json
import os
import argparse
import torch
import random
import numpy as np
import re
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=5, help="Model name")
parser.add_argument("--pass_at", type=int, default=0, help="pass @ how many")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--test_lines", type=int, default=1, help="Number of lines of test code to generate")
parser.add_argument("--num_loops", type=int, default=5, help="Number of times that we do this experiment")
FLAGS = parser.parse_args()

# We will hard-code the stop tokens for llama code family, as the tokenizer is automatically adding start tokens
stop_words = ["\n\n", ("\n","\n")]
assert_stop_words = ["assert"] + stop_words
eof_id = 50256
eof_token = "<|endoftext|>"
imports = "\nimport math\nfrom typing import List\n"
checking_end = f"    pass\n\n# Assume the above function is completed. Write a line of testing code for the function.\n\nassert "


def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    return answer

def trim_starcoder_assert(answer):
    pattern = r',\s*(\"[^\"\\]*(?:\\.[^\"\\]*)*\")?\s*(\r?\n)?\s*$'
    new_answer = re.sub(pattern, '', answer)
    if ".format" in new_answer:
        new_answer = new_answer[:new_answer.find(", \"")]
    return new_answer

def main(args):
    loading_start = time.time()
    number_key = "task_id"
    prompt_key = "text"
    canonical_key = "code"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    num_loops = args.num_loops
    pass_at = args.pass_at
    batch = args.batch
    full_num = 974
    all_time_spent = np.zeros(num_loops)

    # Load HumanEval Dataset
    all_questions_dict = json.load(open("../../evaluations/mbpp/mbpp.json", "r"))

    # Prepare the model checkpoint
    if args.model == 0:
        model_size = "350M"
    elif args.model == 1:
        model_size = "2B"
    elif args.model == 2:
        model_size = "6B"
    elif args.model == 3:
        model_size = "16B"
    checkpoint = f"Salesforce/codegen-{model_size}-mono"
    print(f"Model is {checkpoint}")
    print(f"Pass @ {args.pass_at}")
    print(f"testlines {args.test_lines}")
    print(f"num loops {args.num_loops}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto", padding_side='left')
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

    for loop in range(num_loops):
        selected_numbers = random.sample(range(1, full_num + 1), batch)
        print(f"Selected numbers are {selected_numbers}")
        all_prompts = []
        all_prompts_unique = []
        for sn in selected_numbers:
            question = all_questions_dict[sn-1]
            prompt = question[prompt_key]
            number = question[number_key]
            canonical_code = question[canonical_key]
            prompt = prompt.replace("\r", "")
            canonical_code = canonical_code.replace("\r", "")
            first_def_line = ""
            for line in canonical_code.split('\n'):
                if "def " in line:
                    first_def_line = line
                    break
            prompt += "\n" + first_def_line
            all_prompts = all_prompts + [prompt]*max(pass_at,1)
            all_prompts_unique.append(prompt)
        start = time.time()
        # print(f"length of all_prompts is {len(all_prompts)}")
        # print(all_prompts)
        # print("That is all_prompts")
        # input()
        prompt_ids = tokenizer.batch_encode_plus(all_prompts, padding=True, return_tensors="pt").to(torch.cuda.current_device())
        logits_processor = LogitsProcessorList([StopSequences(stop_words, batch_size=batch*max(pass_at,1), encounters=1)])
        
        # Generate answers
        max_token_num = 550 if number==493 else 350
        with torch.no_grad():
            if pass_at in [0,1]:
                answer_ids = model.generate(
                    **prompt_ids,
                    use_cache = True,
                    pad_token_id = tokenizer.eos_token_id,
                    max_new_tokens = max_token_num,
                    num_beams = 1,
                    do_sample = False,
                    logits_processor = logits_processor
                )
            else:
                answer_ids = model.generate(
                    **prompt_ids,
                    use_cache = True,
                    pad_token_id = tokenizer.eos_token_id,
                    max_new_tokens = max_token_num,
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
        gc.collect()
        
        pass_idx = 0
        if pass_at == 0:
            pass_idx += 1
            this_answer = " " + answer_trimmed[0]
            testcode_trimmed = ""
        else:
            # Generate test code with greedy search. Here we just generate one line, and we pick 
            # the one answer that passes it; otherwise we just use the last one.
            all_prompt_testcode = []
            for prompt in all_prompts_unique:
                first_def_line = ""
                for line in prompt.split('\n'):
                    if "def " in line:
                        first_def_line = line
                        break
                def_name = first_def_line.split("def ")[1].split("(")[0] + "("
                prompt_testcode = prompt + checking_end + def_name
                all_prompt_testcode.append(prompt_testcode)
            # print(f"length of all_prompt_testcode is {len(all_prompt_testcode)}")
            # print(all_prompt_testcode)
            # print("That is all_prompt_testcode")
            prompt_ids = tokenizer.batch_encode_plus(all_prompt_testcode, padding=True, return_tensors="pt").to(torch.cuda.current_device())
            logits_processor = LogitsProcessorList([StopSequences(assert_stop_words, batch_size=batch, encounters=1)])
            with torch.no_grad():
                testcode_ids = model.generate(
                    **prompt_ids,
                    use_cache = True,
                    pad_token_id = tokenizer.eos_token_id,
                    max_new_tokens = 100*args.test_lines,
                    do_sample = False,
                    num_beams = 1,
                    logits_processor = logits_processor
                )
            testcode_ids = testcode_ids[:, len(prompt_ids[0]):]
            testcode_text = tokenizer.batch_decode(testcode_ids, skip_special_tokens=False)
            torch.cuda.empty_cache()
            
        end = time.time()
        # print(f"answer is\n{answer_trimmed}")
        # print(len(answer_trimmed))
        # input()
        # print(f"testcode_text is\n{testcode_text}")
        # print(len(testcode_text))
        # input()
        print(f"time spent {end - start} seconds")
        all_time_spent[loop] = end - start
            
    print(f"{model_size} batch {batch} pass@{pass_at}, average time spent is {np.mean(all_time_spent)} seconds")
            


if __name__== "__main__":
    main(FLAGS)
