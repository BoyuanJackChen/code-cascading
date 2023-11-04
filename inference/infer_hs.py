from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# device = torch.device('cuda:7')
stop_words = ["\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n"]
pe_py = "<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise.\n"
# hello world
# prompt = "\n\ndef hello_world():\n    \"\"\"Write a hello world function in Python3. End with double new line.\n    \"\"\"\n"
#0  (medium)
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
#7  (easy)
prompt_7 = "from typing import List\n\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n"
#31
prompt_31 = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
#35
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
# 155 (hard)
#prompt = "\ndef even_odd_count(num):\n    \"\"\"Given an integer. return a tuple that has the number of even and odd digits respectively.\n\n     Example:\n        even_odd_count(-12) ==> (1, 1)\n        even_odd_count(123) ==> (1, 2)\n    \"\"\"\n"
#161 (hard)
# prompt = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"



if __name__== "__main__":
    checkpoint = "bigcode/starcoder"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    # model.set_target_token_index()
    model.set_want_exited_layers(True)
    # model.set_softmax_threshold(1.0)
    model.set_hidden_states_threshold(0.999)
    # model.set_must_full_token(25)

    for prompt in [prompt_0, prompt_7, prompt_31, prompt_35]:
        prompt += pe_py
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=500,
            num_beam_groups=4,
            diversity_penalty=0.3,
            num_beams=4,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        last_comment_line = "# Here is the correct implementation of the code exercise.\n"
        generated_text = generated_text[0].split(last_comment_line)[-1]
        print(generated_text)

        all_layer_indices = model.get_early_exit_layer_indices().tolist()
        end_index = all_layer_indices.index(-1)
        actual_layer_indices = all_layer_indices[:end_index]
        actual_layer_indices = [int(i) for i in actual_layer_indices]
        print(f"Actual early exit layer indices are \n{actual_layer_indices}; \n"
            f"average is {sum(actual_layer_indices)/len(actual_layer_indices)}; "
            f"length is {len(actual_layer_indices)}")
        
        model.clear_early_exit_layer_indices()

