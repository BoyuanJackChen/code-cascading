from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CodeGenModel
import torch

# device = torch.device('cuda:6')
eos_token = 50256
stop_words = []
# hello world
# prompt = "Write a hello world function in Python3. End with double new line. \ndef hello_world():"
# 0  (easy)
# prompt = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
# 7  (easy)
# prompt = "from typing import List\n\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n"
#31
prompt = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
# 161 (hard)
# prompt = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"

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


if __name__== "__main__":
    # has_close_elements
    # 0 16B
    # answer_list = ['    ', 'for', ' i', ' in', ' range', '(', 'len', '(', 'n', 'umbers', ')', '):', '\n', '        ', 'for', ' j', ' in', ' range', '(', 'i', ' +', ' 1', ',', ' len', '(', 'n', 'umbers', ')', '):', '\n', '            ', 'if', ' abs', '(', 'n', 'umbers', '[', 'i', ']', ' -', ' numbers', '[', 'j', '])', ' <', ' threshold', ':', '\n', '                ', 'return', ' True', '\n', '    ', 'return', ' False', '\n\n']
    # 0 350M
    # answer_list = ['    ', 'for', ' i', ' in', ' range', '(', 'len', '(', 'n', 'umbers', ')', ' -', ' 1', '):', '\n', '        ', 'for', ' j', ' in', ' range', '(', 'i', ' +', ' 1', ',', ' len', '(', 'n', 'umbers', ')', '):', '\n', '            ', 'if', ' abs', '(', 'n', 'umbers', '[', 'i', ']', ' -', ' numbers', '[', 'j', '])', ' <', ' threshold', ':', '\n', '                ', 'return', ' True', '\n', '    ', 'return', ' False', '\n\n']
    # 0 2B
    # answer_list = ['    ', 'for', ' i', ' in', ' range', '(', 'len', '(', 'n', 'umbers', ')', ' -', ' 1', '):', '\n', '        ', 'if', ' numbers', '[', 'i', ' +', ' 1', ']', ' -', ' numbers', '[', 'i', ']', ' <', ' threshold', ':', '\n', '            ', 'return', ' True', '\n', '    ', 'return', ' False', '\n\n']
    # 0 6B
    # answer_list = ['    ', 'for', ' i', ' in', ' range', '(', 'len', '(', 'n', 'umbers', ')', '):', '\n', '        ', 'for', ' j', ' in', ' range', '(', 'i', ' +', ' 1', ',', ' len', '(', 'n', 'umbers', ')', '):', '\n', '            ', 'if', ' abs', '(', 'n', 'umbers', '[', 'i', ']', ' -', ' numbers', '[', 'j', '])', ' <', ' threshold', ':', '\n', '                ', 'return', ' True', '\n', '    ', 'return', ' False', '\n\n']
    # 7 solve
    # answer_list = ['    ', 'return', ' [', 's', ' for', ' s', ' in', ' strings', ' if', ' subst', 'ring', ' in', ' s', ']', '\n\n']
    # 31 350M
    answer_list = ['    ', 'if', ' n', ' ==', ' 1', ':', '\n', '        ', 'return', ' False', '\n', '    ', 'if', ' n', ' ==', ' 2', ':', '\n', '        ', 'return', ' True', '\n', '    ', 'if', ' n', ' %', ' 2', ' ==', ' 0', ':', '\n', '        ', 'return', ' False', '\n', '    ', 'for', ' i', ' in', ' range', '(', '3', ',', ' int', '(', 'math', '.', 'sq', 'rt', '(', 'n', '))', ' +', ' 1', ',', ' 2', '):', '\n', '        ', 'if', ' n', ' %', ' i', ' ==', ' 0', ':', '\n', '            ', 'return', ' False', '\n', '    ', 'return', ' True', '\n', '\n']
    # 31 350M finetuned
    # answer_list = ['    ', 'if', ' n', ' ==', ' 1', ':', '\n', '        ', 'return', ' False', '\n', '    ', 'if', ' n', ' ==', ' 2', ':', '\n', '        ', 'return', ' True', '\n', '    ', 'if', ' n', ' %', ' 2', ' ==', ' 0', ':', '\n', '        ', 'return', ' False', '\n', '    ', 'for', ' i', ' in', ' range', '(', '2', ',', ' int', '(', 'math', '.', 'sq', 'rt', '(', 'n', '))', ' +', ' 1', '):', '\n', '        ', 'if', ' n', ' %', ' i', ' ==', ' 0', ':', '\n', '            ', 'return', ' False', '\n', '    ', 'return', ' True']
    # 161 solve
    # answer_list = ['return', " '", "'.", 'join', '(', 'map', '(', 'lambda', ' x', ':', ' x', '.', 'sw', 'ap', 'case', '()', ' if', ' x', '.', 'is', 'alpha', '()', ' else', ' x', ',', ' s', '))', '\n', '\n']
    # checkpoint = "../checkpoints/350M-hard-late8"
    # num_layers = 20
    checkpoint = "Salesforce/codegen-16B-mono"
    num_layers = 34
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    all_layers = []
    target_0 = ''

    print(f"\n{prompt}\n")
    for target_token in answer_list:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"Generating token '{target_token}'")
        for layer in range(1, num_layers):
            model.set_early_exit_layer(layer)
            generated_ids = model.generate(
                input_ids,
                eos_token_id=eos_token,
                pad_token_id=eos_token,
                use_cache = False,
                max_new_tokens=1,
                num_beam_groups=4,
                diversity_penalty=0.3,
                num_beams=4,
            )
            model.clear_early_exit_layer_indices()
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
            print(f"layer {layer+1}: '{trimmed_text[0]}'")
            if trimmed_text[0] == target_token:
                print(f"{target_token}: {layer+1}")
                all_layers.append(layer+1)
                break

        prompt = prompt + target_token
    print(prompt)
    print(f"all_answers is {answer_list}")
    print(f"all_layers is {all_layers}")
    print(f"mean is {sum(all_layers) / len(all_layers)}")
    for i in range(len(answer_list)):
        print(f"{answer_list[i]}: {all_layers[i]}")
