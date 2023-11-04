from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
import time
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--pass_at", type=int, default=1)
FLAGS = parser.parse_args()

checkpoint = "WizardLM/WizardCoder-3B-V1.0"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
eos_token_id = 0
pe_py = "<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise in python:\n    "

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

stop_words = ["\n\n", ("\n","\n")]
assert_stop_words = ["assert"] + stop_words
print(tokenizer.batch_encode_plus(assert_stop_words)['input_ids'])
print(tokenizer.encode("<|endoftext|>"))
# 0
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
# 31
prompt_31 = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
# 35
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
# 161
prompt_161 = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"


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


def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    trimmed = False
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        answer = answer.lstrip('\n')
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
                    trimmed = True
        answer = answer[:min_i]
        result.append(answer)
    if not trimmed:
        print("This question is not trimmed!")
    return result


def main(args):
    # Initialize model
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")
    
    # Generate the selected prompts one at a time
    for prompt in [prompt_0, prompt_31, prompt_35, prompt_161]:
        # Find the line that starts with "def"
        prompt_lines = prompt.split("\n")
        def_line = ""
        for i, line in enumerate(prompt_lines):
            if line.startswith("def"):
                def_line = line
                break
        prompt += "\n" + pe_py + def_line + '\n    """\n    Do not generate any comment\n    """\n    '
        input_ids = tokenizer.batch_encode_plus([prompt]*args.pass_at, return_tensors="pt").to(torch.cuda.current_device())
        logits_processor = LogitsProcessorList([StopSequences(stop_words, batch_size=args.pass_at, encounters=1)])
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
                # logits_processor = logits_processor
            )
        print(f"Time to generate is {time.time() - start_generating}")
        generated_ids = generated_ids[:, len(input_ids['input_ids'][0]):]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        generated_len = len(decoded_list)
        print(f"per token time is {(time.time()-start_generating)/generated_len}")
        print(f"\nGenerated ids is:\n{generated_ids[0]}")

        # Print text
        # print(f"\nDecoded_list is:\n{decoded_list}")
        print(f"\ngenerated_text is:\n{generated_text[0]}")

if __name__== "__main__":
    main(FLAGS)