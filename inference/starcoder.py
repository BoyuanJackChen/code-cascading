from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch

stopwords = ["\n\n\n", "\n\n"]
pe_py = "<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise in python:\n"
pe_cpp = "<filename>solutions/solution_1.cpp\n// Here is the correct implementation of the code exercise in c++:\n"
pe_jv = "<filename>solutions/solution_1.java\n// Here is the correct implementation of the code exercise in java:\n"
pe_js = "<filename>solutions/solution_1.js\n// Here is the correct implementation of the code exercise in javascript:\n"
pe_go = "<filename>solutions/solution_1.go\n// Here is the correct implementation of the code exercise in go:\n"
feedback_unspecific = "# Above is the answer you just generated, but it was actually incorrect. Can you fix it?\n"

#0
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
#1
prompt_1 = "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"
#31
prompt_31 = "\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
#35
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"

# Java doesn't need the pe
# 0 - Java
prompt_j0 = "import java.util.*;\nimport java.lang.*;\n\nclass Solution {\n    /**\n    Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> hasCloseElements(Arrays.asList(1.0, 2.0, 3.0), 0.5)\n    false\n    >>> hasCloseElements(Arrays.asList(1.0, 2.8, 3.0, 4.0, 5.0, 2.0), 0.3)\n    true\n     */\n    public boolean hasCloseElements(List<Double> numbers, double threshold) {\n"
# 1 - Java
prompt_j1 = "import java.util.*;\nimport java.lang.*;\n\nclass Solution {\n    /**\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separateParenGroups(\"( ) (( )) (( )( ))\")\n    [\"()\", \"(())\", \"(()())\"]\n     */\n    public List<String> separateParenGroups(String paren_string) {\n"
# 31 - Java
prompt_j31 = "import java.util.*;\nimport java.lang.*;\n\nclass Solution {\n    /**\n    Return true if a given number is prime, and false otherwise.\n    >>> isPrime(6)\n    false\n    >>> isPrime(101)\n    true\n    >>> isPrime(11)\n    true\n    >>> isPrime(13441)\n    true\n    >>> isPrime(61)\n    true\n    >>> isPrime(4)\n    false\n    >>> isPrime(1)\n    false\n     */\n    public boolean isPrime(int n) {\n"

# C++ needs the pe
# 0 - C++
prompt_c0 = "/*\nCheck if in given vector of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> has_close_elements({1.0, 2.0, 3.0}, 0.5)\nfalse\n>>> has_close_elements({1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)\ntrue\n*/\n#include<stdio.h>\n#include<vector>\n#include<math.h>\nusing namespace std;\nbool has_close_elements(vector<float> numbers, float threshold){\n"
# 1 - C++
prompt_c1 = "/*\nInput to this function is a string containing multiple groups of nested parentheses. Your goal is to\nseparate those group into separate strings and return the vector of those.\nSeparate groups are balanced (each open brace is properly closed) and not nested within each other\nIgnore any spaces in the input string.\n>>> separate_paren_groups(\"( ) (( )) (( )( ))\")\n{\"()\", \"(())\", \"(()())\"}\n*/\n#include<stdio.h>\n#include<vector>\n#include<string>\nusing namespace std;\nvector<string> separate_paren_groups(string paren_string){\n"
# 31 - C++
prompt_c31 = "/*\nReturn true if a given number is prime, and false otherwise.\n>>> is_prime(6)\nfalse\n>>> is_prime(101)\ntrue\n>>> is_prime(11)\ntrue\n>>> is_prime(13441)\ntrue\n>>> is_prime(61)\ntrue\n>>> is_prime(4)\nfalse\n>>> is_prime(1)\nfalse\n*/\n#include<stdio.h>\nusing namespace std;\nbool is_prime(long long n){\n"

# JS needs the pe, specifying it is in JavaScript
# 0 - JS
prompt_js0 = "/* Check if in given list of numbers, are any two numbers closer to each other than\n  given threshold.\n  >>> hasCloseElements([1.0, 2.0, 3.0], 0.5)\n  false\n  >>> hasCloseElements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n  true\n  */\nconst hasCloseElements = (numbers, threshold) => {\n"
# 1 - JS
prompt_js1 = "/* Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n  separate those group into separate strings and return the list of those.\n  Separate groups are balanced (each open brace is properly closed) and not nested within each other\n  Ignore any spaces in the input string.\n  >>> separateParenGroups('( ) (( )) (( )( ))')\n  ['()', '(())', '(()())']\n  */\nconst separateParenGroups = (paren_string) => {\n"
# 31 - JS
prompt_js31 = "/*Return true if a given number is prime, and false otherwise.\n  >>> isPrime(6)\n  false\n  >>> isPrime(101)\n  true\n  >>> isPrime(11)\n  true\n  >>> isPrime(13441)\n  true\n  >>> isPrime(61)\n  true\n  >>> isPrime(4)\n  false\n  >>> isPrime(1)\n  false\n  */\nconst isPrime = (n) => {\n"

# Go needs the pe
# 0 - Go
prompt_go0 = "import (\n    \"math\"\n)\n\n// Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n// >>> HasCloseElements([]float64{1.0, 2.0, 3.0}, 0.5)\n// false\n// >>> HasCloseElements([]float64{1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)\n// true\nfunc HasCloseElements(numbers []float64, threshold float64) bool {\n"
# 1 - Go
prompt_go1 = "\n// Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n// separate those group into separate strings and return the list of those.\n// Separate groups are balanced (each open brace is properly closed) and not nested within each other\n// Ignore any spaces in the input string.\n// >>> SeparateParenGroups('( ) (( )) (( )( ))')\n// ['()', '(())', '(()())']\nfunc SeparateParenGroups(paren_string string) []string {\n"
# 31 - Go
prompt_go31 = "\n// Return true if a given number is prime, and false otherwise.\n// >>> IsPrime(6)\n// false\n// >>> IsPrime(101)\n// true\n// >>> IsPrime(11)\n// true\n// >>> IsPrime(13441)\n// true\n// >>> IsPrime(61)\n// true\n// >>> IsPrime(4)\n// false\n// >>> IsPrime(1)\n// false\nfunc IsPrime(n int) bool {\n"


# Fixes
prompt_1_bad = '''
def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # Initialize a list to store the groups
    groups = []
    # Initialize a variable to store the current group
    current_group = ''
    # Initialize a variable to store the number of open braces
    open_braces = 0
    # Loop through each character in the input string
    for char in paren_string:
        # If the character is an open brace, increment the number of open braces
        if char == '(':
            open_braces += 1
        # If the character is a closed brace, decrement the number of open braces
        elif char == ')':
            open_braces -= 1
        # If the character is a space, ignore it
        elif char =='':
            continue
        # If the number of open braces is 0, that means we have reached the end of a group
        if open_braces == 0:
            # Append the current group to the list of groups
            groups.append(current_group)
            # Reset the current group and the number of open braces
            current_group = ''
            open_braces = 0
        # Otherwise, add the character to the current group
        else:
            current_group += char
    # Return the list of groups
    return groups
'''

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        answer = answer.split('"""')[-1]
        answer = answer.lstrip('\n')
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
        answer = answer[:min_i]
        result.append(answer)
    return result

if __name__== "__main__":
    checkpoint = "bigcode/starcoder"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")
    model.eval()

    for prompt in [prompt_1_bad]:
        # prompt = prompt_1 + pe_py + prompt
        prompt += feedback_unspecific + pe_py
        # prompt += pe_py
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        start_generating = time.time()
        generated_ids = model.generate(
            input_ids,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            # attention_mask=torch.zeros(1).cuda(),
            max_new_tokens=400,
            # min_new_tokens=300,
            # do_sample = False,
            # num_beams=1,
            # num_beam_groups=1,
            num_beams=4,
            num_beam_groups=4,
            diversity_penalty=0.3,
            # do_sample=True
            # top_k = 20,
            # do_sample = False,
            # penalty_alpha = 0.3
            # stopping_criteria=stopping_criteria,
        )
        print(f"Time to generate is {time.time() - start_generating}")
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        print(decoded_list)
        generated_len = len(decoded_list) - len(input_ids[0])
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt = tokenizer.decode(prompt_ids[0])
        trimmed_text = trim_with_stopwords(generated_text, stopwords, prompt)
        print(f"\ngenerated_text is:\n{generated_text[0]}")
