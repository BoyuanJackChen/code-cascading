from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="Salesforce/codegen-350M-mono", help="Model path")
FLAGS = parser.parse_args()

# Codegen: 350M, 2B, 6B, 16B
# Starcoder: 16B

eos_token = 50256
stop_words = ["\n\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n", "\n\n"]
# prompt = "Write a hello world function in Python3. End with double new line. \ndef hello_world():"
# 0 - easy
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
# 31
prompt_31 = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
# 35
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
# 161
prompt_161 = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"

# Go 47
prompt_go47 = "import (\n    \"sort\"\n)\n\n// Return Median of elements in the list l.\n// >>> Median([3, 1, 2, 4, 5])\n// 3.0\n// >>> Median([-10, 4, 6, 1000, 10, 20])\n// 15.0\nfunc Median(l []int) float64 {\n"
# Go 48
prompt_go48 = "\n// Checks if given string is a palindrome\n// >>> IsPalindrome('')\n// true\n// >>> IsPalindrome('aba')\n// true\n// >>> IsPalindrome('aaaaa')\n// true\n// >>> IsPalindrome('zbcd')\n// false\nfunc IsPalindrome(text string) bool {\n"

# Checking prompt
checking_start = "Write a set of testing code for the following Python function:\n"
checking_end = "pass\n\nAssume the above function is completed. Write a set of testing code for the function.\n\nassert"

# Correctness 0
prompt_correct0 = '''
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False'''

prompt_1_redit = '''from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """
    Given a string of balanced, non-overlapping parentheses groups, this function separates each group and returns them as a list. 
    It ignores spaces within the input string. 
    Each group of parentheses can be nested.
    
    Example:
    separate_paren_groups('( ) (( )) (( )( ))') should return ['()', '(())', '(()())']
    """
    '''

prompt_7 = "from typing import List\n\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n"
prompt_7 += "    return [x for x in strings if substring in x]"
# prompt_7 += "return ['aha']"

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
    # Initialize model and tokenizer
    checkpoint = "Salesforce/codegen-6B-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto", eos_token_id=[50256, 628])
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")

    # # Log model stats
    # print(type(model))
    # print(f"vocab size is {model.get_vocab_size()}")
    # print(f"input embeddings is: {model.get_input_embeddings()}")
    # print(f"embed dimension is: {model.get_embed_dim()}")
    # print(f"dropout is {model.get_dropout()}")
    # # print(f"h is {model.get_h()}")
    # print(f"rotary_dim is {model.get_rotary_dim()}")
    # print(f"layer norm is {model.get_ln_f()}")
    # print(f"lm head is {model.get_lm_head()}")
    
    for prompt in [prompt_correct0]:
        prompt = prompt + checking_end
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        start_generating = time.time()
        generated_ids = model.generate(
            input_ids,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            # attention_mask=torch.zeros(1).cuda(),
            max_length=300,
            # max_new_tokens=200,
            # min_new_tokens=200,
            do_sample = False,
            # temperature = 1.0,
            # top_k = 0,
            # top_p = 1.0,
            # num_beams=1,
            # num_beam_groups=1,
            num_beams=4,
            num_beam_groups=4,
            diversity_penalty=0.3,
            # top_k = 20,
            # penalty_alpha = 0.3
            # stopping_criteria=stopping_criteria,
        )
        print(f"Time to generate is {time.time() - start_generating}")
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        generated_len = len(decoded_list) - len(input_ids[0])
        # print(f"generated_ids is {generated_ids[0]}")
        print(f"decoded_list is {decoded_list}")
        # print(f"decoded list has length {len(decoded_list)}; prompt has length {len(input_ids[0])};"
        #       f"generated has length {generated_len}")
        print(f"per token time is {(time.time()-start_generating)/generated_len}")

        # Print text
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt = tokenizer.decode(prompt_ids[0])
        trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
        print(f"\ngenerated_text is:\n{generated_text[0]}")
        # print("trimmed_text is:")
        # print(trimmed_text[0])

if __name__== "__main__":
    main(FLAGS)