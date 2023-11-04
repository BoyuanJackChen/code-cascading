from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
import time
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--pass_at", type=int, default=42)
FLAGS = parser.parse_args()

checkpoint = "WizardLM/WizardCoder-Python-7B-V1.0"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")

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

# We will hard-code the stop tokens for llama code family, as the tokenizer is automatically adding start tokens
stop_words = ["\n\n", ("\n","\n")]
stop_words_ids = [[13,13]]
assert_stop_words = ["assert"] + stop_words
assert_stop_words_ids = [[9294]] + stop_words_ids
eof_id = 2
eof_token = "</s>"
imports = "\nimport math\nfrom typing import List\n"

# 0
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
# 1
prompt_1 = "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"
# 31
prompt_31 = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
# 35
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
# 161
prompt_161 = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"


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
    model.eval()
    print(f"Time to load model is {time.time() - start_load_model}")
    pass_at = args.pass_at
    
    # Stopping criteria for generation using the LogitsProcessor class
    class StopSequences(LogitsProcessor):
        def __init__(self, stop_sequences, batch_size, encounters=1, eos_token_id=2, test_lines=1):
            StoppingCriteria.__init__(self)
            self.stop_sequences = assert_stop_words_ids
            self.batch_size = batch_size
            self.encounters = [encounters] * batch_size
            self.NUM_ENCOUNTERS = encounters
            self.eos_token_id = eos_token_id
            self.just_started = True
            self.test_lines = test_lines - 1

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
                        if stop == self.stop_sequences[0] and self.test_lines > 0:
                            self.test_lines -= 1
                            continue
                        self.encounters[i] -= 1
                        if self.encounters[i] <= 0:
                            scores[i] = forced_eos
            return scores
    
    # Generate the selected prompts one at a time
    for prompt in [prompt_0, prompt_1, prompt_31, prompt_35, prompt_161]:
        final_array = [prompt]*pass_at
        input_ids = tokenizer.batch_encode_plus(final_array, return_tensors="pt").to(torch.cuda.current_device())
        logits_processor = LogitsProcessorList([StopSequences(assert_stop_words_ids, batch_size=pass_at, encounters=1, test_lines=1)])
        start_generating = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **input_ids,
                use_cache = True,
                pad_token_id = tokenizer.eos_token_id,
                max_new_tokens = 310,
                do_sample = False,
                # temperature = 0.0,
                num_beams = 1,
                # num_return_sequences = args.pass_at,
                logits_processor = logits_processor
            )
        print(f"Time to generate is {time.time() - start_generating}")
        torch.cuda.empty_cache()

if __name__== "__main__":
    main(FLAGS)