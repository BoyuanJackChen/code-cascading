from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch

eos_token = 0
stop_words = ["\n\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n", "\n\n"]
instruction = "Complete the Python function given the prompt below. Do not copy the definition line, just start" \
              "with the body of the function.\n"
# prompt = "Write a hello world function in Python3. End with double new line. \ndef hello_world():"
#0
# prompt = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
#35
# prompt = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
#31
prompt = "\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
#161
# prompt = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"
# LC94, easy
# prompt = "class Solution:\n    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:\n        \"\"\"\n        Given the root of a binary tree, return the inorder traversal of its nodes' values.\n        Example 1:\n        Input: root = [1,null,2,3]\n        Output: [1,3,2]\n        Example 2:\n        Input: root = []\n        Output: []\n        Example 3:\n        Input: root = [1]\n        Output: [1]\n        \"\"\"\n"
# LC97, medium
# prompt = "class Solution:\n    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:\n\t\t\"\"\"\n\t\tGiven strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.\n\t\tAn interleaving of two strings s and t is a configuration where s and t are divided into n and m substrings respectively, such that:\n\t\t\ts = s1 + s2 + ... + sn\n\t\t\tt = t1 + t2 + ... + tm\n\t\t\t|n - m| <= 1\n\t\t\tThe interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...\n\t\tNote: a + b is the concatenation of strings a and b.\n\t\tExample 1:\n\t\tInput: s1 = \"aabcc\", s2 = \"dbbca\", s3 = \"aadbbcbcac\"\n\t\tOutput: true\n\t\tExplanation: One way to obtain s3 is:\n\t\tSplit s1 into s1 = \"aa\" + \"bc\" + \"c\", and s2 into s2 = \"dbbc\" + \"a\".\n\t\tInterleaving the two splits, we get \"aa\" + \"dbbc\" + \"bc\" + \"a\" + \"c\" = \"aadbbcbcac\".\n\t\tSince s3 can be obtained by interleaving s1 and s2, we return true.\n\t\tExample 2:\n\t\tInput: s1 = \"aabcc\", s2 = \"dbbca\", s3 = \"aadbbbaccc\"\n\t\tOutput: false\n\t\tExplanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.\n\t\tExample 3:\n\t\tInput: s1 = \"\", s2 = \"\", s3 = \"\"\n\t\tOutput: true\n\t\t\"\"\"\n"
# LC976, easy
# prompt = "class Solution:\n    def largestPerimeter(self, nums: List[int]) -> int:\n\t\t\"\"\"\n\t\tGiven an integer array nums, return the largest perimeter of a triangle with a non-zero area, formed from three of these lengths. If it is impossible to form any triangle of a non-zero area, return 0.\n\t\tExample 1:\n\t\tInput: nums = [2,1,2]\n\t\tOutput: 5\n\t\tExplanation: You can form a triangle with three side lengths: 1, 2, and 2.\n\t\tExample 2:\n\t\tInput: nums = [1,2,1,10]\n\t\tOutput: 0\n\t\tExplanation: \n\t\tYou cannot use the side lengths 1, 1, and 2 to form a triangle.\n\t\tYou cannot use the side lengths 1, 1, and 10 to form a triangle.\n\t\tYou cannot use the side lengths 1, 2, and 10 to form a triangle.\n\t\tAs we cannot use any three side lengths to form a triangle of non-zero area, we return 0.\n\t\t\"\"\"\n"
# LC962, medium
# prompt = "class Solution:\n    def maxWidthRamp(self, nums: List[int]) -> int:\n\t\t\"\"\"\n\t\tA ramp in an integer array nums is a pair (i, j) for which i < j and nums[i] <= nums[j]. The width of such a ramp is j - i.\n\t\tGiven an integer array nums, return the maximum width of a ramp in nums. If there is no ramp in nums, return 0.\n\t\tExample 1:\n\t\tInput: nums = [6,0,8,2,1,5]\n\t\tOutput: 4\n\t\tExplanation: The maximum width ramp is achieved at (i, j) = (1, 5): nums[1] = 0 and nums[5] = 5.\n\t\tExample 2:\n\t\tInput: nums = [9,8,1,0,1,9,4,0,4,1]\n\t\tOutput: 7\n\t\tExplanation: The maximum width ramp is achieved at (i, j) = (2, 9): nums[2] = 1 and nums[9] = 1.\n\t\t\"\"\"\n",

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    trimmed = False
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt*2-10:]
        for i in range(len(answer)):
            if answer[i:].startswith('"""\n'):
                answer = answer[i+4:]
                break
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


if __name__== "__main__":
    print(prompt)
    prompt = prompt.replace("\t", "    ")
    all_checkpoints = ["bigcode/starcoder", "Salesforce/codegen-350M-mono", "Salesforce/codegen-2B-mono",
                       "Salesforce/codegen-6B-mono", "Salesforce/codegen-16B-mono"]
    for checkpoint in all_checkpoints:
        print(f"Working on model {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
        start_load_model = time.time()
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32, device_map="auto")
        print(f"Time to load model is {time.time() - start_load_model}")
        model.eval()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        start_generating = time.time()
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=150,
            # min_new_tokens=200,
            num_beams=4,
            num_beam_groups=4,
            diversity_penalty=0.3,
        )
        print(f"Time to generate is {time.time() - start_generating}")
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        # print(f"decoded_list is {decoded_list}")
        generated_len = len(decoded_list) - len(input_ids[0])
        # print(f"decoded list has length {len(decoded_list)}; prompt has length {len(input_ids[0])};"
        #       f"generated has length {generated_len}")
        print(f"per token time is {(time.time()-start_generating)/generated_len}")
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt = tokenizer.decode(prompt_ids[0])
        trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
        print(f"\ngenerated_text is:\n{generated_text[0]}")
        # print("trimmed_text is:")
        # print(trimmed_text[0])
