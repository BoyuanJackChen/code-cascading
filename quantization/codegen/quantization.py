from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
import time
import json
import os
import argparse
import multiprocessing
import torch
from human_eval.data import write_jsonl, read_problems, stream_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=0, help="Model name")
parser.add_argument("--pass_at", type=int, default=0, help="pass @ how many")
parser.add_argument("--num_loops", type=int, default=0, help="Number of times that we do this experiment")
FLAGS = parser.parse_args()

# We will hard-code the stop tokens for llama code family, as the tokenizer is automatically adding start tokens
stop_words = ["\n\n", ("\n","\n")]
assert_stop_words = ["assert"] + stop_words
eos_id = 50256
eos_token = "<|endoftext|>"
imports = "\nimport math\nfrom typing import List\n"

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    return answer

def process_answer(answer):
    answer = answer[:answer.find("\n#")]
    answer = answer.replace("\r", "")
    answer = answer.replace("\t", "    ")
    answer = trim_substring_from_end(answer, eos_token)
    answer = trim_substring_from_end(answer, "#")
    answer = trim_substring_from_end(answer, "\n\n")
    return answer


def main(args):
    loading_start = time.time()
    number_key = "task_id"
    prompt_key = "prompt"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pass_at = args.pass_at
    num_loops = args.num_loops if pass_at>1 else 1
    
    # Load HumanEval Dataset
    all_questions_dict = read_problems()
    all_keys = all_questions_dict.keys()

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
    if not os.path.exists(f"answer"):
        os.mkdir(f"answer")
    if not os.path.exists(f"answer/{model_size}"):
        os.mkdir(f"answer/{model_size}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token_id = tokenizer.eos_token_id
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
        output_file_name = f'answer/{model_size}/{model_size}_p{pass_at}_l{loop}.json'
        max_seen_number = -1
        if os.path.exists(output_file_name):
            if os.path.exists(f'{model_size}/{model_size}_p{pass_at}_l{loop+1}.json'):
                continue
            else:
                last_generated_data = json.load(open(output_file_name, "r"))
                for answer_dict in last_generated_data:
                    if answer_dict["number"] > max_seen_number:
                        max_seen_number = answer_dict["number"]
        
        # Go through each question
        total_start = time.time()
        for question_key in all_keys:
            question = all_questions_dict[question_key]
            number = int(question[number_key].split("/")[1])
            if number <= max_seen_number:
                continue
            print(f"On question {number}")
            prompt = question[prompt_key]
            prompt = prompt.replace('    ', '\t')
            prompt_ids = tokenizer.batch_encode_plus([prompt]*max(pass_at,1), return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
            logits_processor = LogitsProcessorList([StopSequences(stop_words, batch_size=max(pass_at,1), encounters=1)])
            
            # Generate answers
            max_new_tokens = 1024
            max_length = 2048
            with torch.no_grad():
                if pass_at in [0,1]:
                    answer_ids = model.generate(
                        **prompt_ids,
                        use_cache = True,
                        pad_token_id = tokenizer.pad_token_id,
                        eos_token_id = tokenizer.eos_token_id,
                        max_new_tokens = max_new_tokens,
                        num_return_sequences=1,
                        do_sample = False,
                        logits_processor = logits_processor
                    )
                else:
                    answer_ids = model.generate(
                        **prompt_ids,
                        use_cache = True,
                        pad_token_id = tokenizer.eos_token_id,
                        eos_token_id = tokenizer.eos_token_id,
                        max_new_tokens = max_new_tokens,
                        do_sample = True,
                        top_k = 0,
                        top_p = 0.95,
                        temperature = 0.8,
                        num_beams = 1,
                        logits_processor = logits_processor
                    )
            answer_ids = answer_ids[:, len(prompt_ids['input_ids'][0]):]
            answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
            answer_trimmed = [process_answer(answer) for answer in answer_text]
            torch.cuda.empty_cache()
            
            for pass_idx, answer in enumerate(answer_trimmed):
                answer_dict = {
                    "number": number,
                    # "prompt": prompt,
                    "checkpoint": model_size,
                    "pass": pass_idx,
                    "answer": answer
                }
                answer_dict_list.append(answer_dict)
                counter += 1

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
        print(f"Total time consumed: {time.time() - total_start}")
                

if __name__== "__main__":
    main(FLAGS)
