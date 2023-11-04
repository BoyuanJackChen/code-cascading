from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
import time
import argparse
import torch
import os
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--pass_at", type=int, default=1)
FLAGS = parser.parse_args()

checkpoint = "WizardLM/WizardCoder-Python-34B-V1.0"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# WizardCoder all checkpoints (ranked by performance):
# WizardLM/WizardCoder-Python-34B-V1.0
# WizardLM/WizardCoder-Python-13B-V1.0
# WizardLM/WizardCoder-Python-7B-V1.0
# WizardLM/WizardCoder-15B-V1.0
# WizardLM/WizardCoder-3B-V1.0
# WizardLM/WizardCoder-1B-V1.0

# Code LLAMA 2
# codellama/CodeLlama-7b-hf
# codellama/CodeLlama-13b-hf
# codellama/CodeLlama-34b-hf

# Salesforce Codegen
# Salesforce/codegen-350M-mono
# Salesforce/codegen-2B-mono
# Salesforce/codegen-6B-mono
# Salesforce/codegen-16B-mono

eos_token = 50256
stop_words = ["\n\n", ("\n","\n"), "</code>", "END SOLUTION"]
stop_words_ids = [[13,13], [829, 401, 29958], [11056, 317, 5607, 2692, 2725], [11794, 317, 5607, 2692, 2725]]
assert_stop_words = ["assert"] + stop_words
assert_stop_words_ids = [[9294]] + stop_words_ids
eof_id = 2
eof_token = "</s>"
imports = "\nimport math\nfrom typing import List\n"
checking_end = "\n# Assume that the above code is completed. Please write a test for the question.\nassert "

def process_ds_prompt(prompt):
    prompt = prompt.replace("\n\n\n", "\n")
    prompt = prompt.replace("\n\n", "\n")
    # prompt cut off the first line.
    prompt = prompt[prompt.find("\n") + 1:]
    prompt = "Complete the python program given the prompt below:\n" + prompt
    # Get the good half-line
    preserved_prompt = prompt[prompt.rfind("</code>"):]
    preserved_prompt = preserved_prompt[preserved_prompt.find("\n") + 1:]
    preserved_prompt = preserved_prompt[:preserved_prompt.find("\n")]
    # Trim the prompt off from the last occurrence of "</code>"
    prompt = prompt[:prompt.find("</code>")]
    # replace "A:\n<code>" with "\nCode:"
    prompt = prompt.replace("A:\n<code>", "\nCode:")
    # prompt += "\n" + preserved_prompt
    return prompt

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

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    if answer.endswith("\n"):
        answer = answer[:-1]
    return answer


def main(args):
    # Initialize model
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")
    
    pass_at = 1
    
    # Load DS-1000 dataset
    all_ds1000_dir = "../evaluations/ds_1000/ds1000_data"
    all_libraries = ["Matplotlib", "Numpy", "Pandas", "Pytorch", "Scipy", "Sklearn", "Tensorflow"]
    library = "Sklearn"
    library_dir = f"{all_ds1000_dir}/{library}/Completion"
    num_questions = len([name for name in os.listdir(library_dir) if os.path.isdir(os.path.join(library_dir, name))])
    print(f"num_questions is {num_questions}")
    for number in range(num_questions):
        # if number < 30:
        #     continue
        question_dir = library_dir + f"/q{number}/"
        with open(question_dir + "prompt.txt", 'r') as f:
            prompt = f.read()
        # Trim prompt from the first "<code>" to the first "</code>"
        if "<code>" in prompt and "</code>" in prompt:
            code_body = prompt[prompt.find("<code>")+len("<code>"):prompt.find("</code>")]
        else:
            code_body = prompt
        print(prompt)
        input_ids = tokenizer.batch_encode_plus([prompt]*max(pass_at,1), return_tensors="pt").to(torch.cuda.current_device())
        logits_processor = LogitsProcessorList([StopSequences(stop_words_ids, batch_size=max(pass_at,1), encounters=1)])
        start_generating = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **input_ids,
                use_cache = True,
                pad_token_id = tokenizer.eos_token_id,
                max_new_tokens = 300,
                do_sample = False,
                # temperature = 0.0,
                num_beams = 1,
                logits_processor = logits_processor,
            )
        # print(f"Time to generate is {time.time() - start_generating}")
        generated_ids = generated_ids[:, len(input_ids['input_ids'][0]):]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        generated_len = len(decoded_list)
        torch.cuda.empty_cache()
        # print(f"\nGenerated ids is:\n{generated_ids[0]}")

        # Print text
        # print(f"\nDecoded_list is:\n{decoded_list}")
        # print(f"\ngenerated_text is:\n{generated_text[0]}")
        answer = trim_substring_from_end(generated_text[0], "</s>")
        answer = trim_substring_from_end(answer, "</code>")
        prompt_testcode = prompt + checking_end
        input_ids = tokenizer.batch_encode_plus([prompt_testcode], return_tensors="pt").to(torch.cuda.current_device())
        logits_processor = LogitsProcessorList([StopSequences(assert_stop_words_ids, batch_size=1, encounters=1)])
        with torch.no_grad():
            testcode_ids = model.generate(
                **input_ids,
                use_cache = True,
                pad_token_id = tokenizer.eos_token_id,
                max_new_tokens = 200,
                do_sample = False,
                # temperature = 0.0,
                num_beams = 1,
                logits_processor = logits_processor,
            )
        testcode_ids = testcode_ids[:, len(input_ids['input_ids'][0]):]
        testcode_text = tokenizer.batch_decode(testcode_ids, skip_special_tokens=False)
        testcode_text = trim_substring_from_end(testcode_text[0], "</s>")
        testcode_text = trim_substring_from_end(testcode_text, "</code>")
        testcode_text = trim_substring_from_end(testcode_text, "assert")
        testcode_text = "assert " + testcode_text
        
        full_code = code_body + "\n# answer:\n" + answer + "\n\n" + "# test:\n" + testcode_text
        print(f"full_code:\n{full_code}")
        def code_to_run(result_queue):
            try:
                exec(full_code, globals())
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
        print(correct)
        input()
if __name__== "__main__":
    main(FLAGS)