from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="350M")
FLAGS = parser.parse_args()

eos_token = 50256
stop_words = ["\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n"]
# prompt = "Write a hello world function in Python3. End with double new line. \ndef hello_world():"
# 0  (medium)
prompt = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
# 7  (easy)
# prompt = "from typing import List\n\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n"
# 35
# prompt = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
# 31
# prompt = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
# 155
# prompt = "\ndef even_odd_count(num):\n    \"\"\"Given an integer. return a tuple that has the number of even and odd digits respectively.\n\n     Example:\n        even_odd_count(-12) ==> (1, 1)\n        even_odd_count(123) ==> (1, 2)\n    \"\"\"\n"

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
        answer = answer[:min_i]
        result.append(answer)
    return result


def main(args):
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/codegen-{model_name}-mono", device_map="auto")
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(f"Salesforce/codegen-{model_name}-mono", torch_dtype=torch.float32, device_map="auto")
    load_time = time.time() - load_start
    print(f"Model {model_name} loading time is {load_time} seconds")
    model.eval()
    print(f"\n{prompt}\n")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    for _ in range(3):
        gen_start = time.time()
        generated_ids = model.generate(
            input_ids,
            use_cache = True,
            do_sample = True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=300,
            num_beams=1,
            # num_beam_groups=4,
            # diversity_penalty=0.3,
            temperature=0.2,
            top_p = 0.95,
            top_k = 20,
        )
        gen_time = time.time() - gen_start
        print(f"generated_ids has length {len(generated_ids[0])}; input_ids has length {len(input_ids[0])}; "
            f"generation time is {gen_time} seconds")
        print(f"per-token generation time is {gen_time / (len(generated_ids[0])-len(input_ids[0]))} seconds")
        print()
        

        # Find where the actual output is generated / Codegen forward is called (utils.py). and apply to an earlier layer.
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
        # print(f"below is trimmed text:")
        # print(trimmed_text[0])

if __name__== "__main__":
    main(FLAGS)
