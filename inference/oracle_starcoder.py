from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CodeGenModel
import torch

# 0
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
# 1
prompt_1 = "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
# 31
prompt_31 = "\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
# 35
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
prompt_array = [prompt_0, prompt_1, prompt_31, prompt_35]


if __name__== "__main__":
    # 0
    answer_list_0 = ['\n', 'def', ' has', '_', 'close', '_', 'elements', '(', 'numbers', ':', ' List', '[', 'float', '],', ' threshold', ':', ' float', ')', ' ->', ' bool', ':', '\n   ', ' """', ' Check', ' if', ' in', ' given', ' list', ' of', ' numbers', ',', ' are', ' any', ' two', ' numbers', ' closer', ' to', ' each', ' other', ' than', '\n   ', ' given', ' threshold', '.', '\n   ', ' >>>', ' has', '_', 'close', '_', 'elements', '([', '1', '.', '0', ',', ' ', '2', '.', '0', ',', ' ', '3', '.', '0', '],', ' ', '0', '.', '5', ')', '\n   ', ' False', '\n   ', ' >>>', ' has', '_', 'close', '_', 'elements', '([', '1', '.', '0', ',', ' ', '2', '.', '8', ',', ' ', '3', '.', '0', ',', ' ', '4', '.', '0', ',', ' ', '5', '.', '0', ',', ' ', '2', '.', '0', '],', ' ', '0', '.', '3', ')', '\n   ', ' True', '\n   ', ' """', '\n   ', ' for', ' i', ' in', ' range', '(', 'len', '(', 'numbers', ')):', '\n       ', ' for', ' j', ' in', ' range', '(', 'i', ' +', ' ', '1', ',', ' len', '(', 'numbers', ')):', '\n           ', ' if', ' abs', '(', 'numbers', '[', 'i', ']', ' -', ' numbers', '[', 'j', '])', ' <', ' threshold', ':', '\n               ', ' return', ' True', '\n   ', ' return', ' False', '\n', '<|endoftext|>']
    # 1
    answer_list_1 = ['\n', 'def', ' separate', '_', 'paren', '_', 'groups', '(', 'paren', '_', 'string', ':', ' str', ')', ' ->', ' List', '[', 'str', ']:', '\n   ', ' """', ' Input', ' to', ' this', ' function', ' is', ' a', ' string', ' containing', ' multiple', ' groups', ' of', ' nested', ' parentheses', '.', ' Your', ' goal', ' is', ' to', '\n   ', ' separate', ' those', ' group', ' into', ' separate', ' strings', ' and', ' return', ' the', ' list', ' of', ' those', '.', '\n   ', ' Separ', 'ate', ' groups', ' are', ' bal', 'anced', ' (', 'each', ' open', ' br', 'ace', ' is', ' properly', ' closed', ')', ' and', ' not', ' nested', ' within', ' each', ' other', '\n   ', ' Ignore', ' any', ' spaces', ' in', ' the', ' input', ' string', '.', '\n   ', ' >>>', ' separate', '_', 'paren', '_', 'groups', "('", '(', ' )', ' ((', ' ))', ' ((', ' )(', ' ))', "')", '\n   ', " ['", "()',", " '(", '())', "',", " '", '(()', '())', "']", '\n   ', ' """', '\n   ', ' #', ' Initialize', ' a', ' list', ' to', ' store', ' the', ' groups', '\n   ', ' groups', ' =', ' []', '\n   ', ' #', ' Initialize', ' a', ' variable', ' to', ' store', ' the', ' current', ' group', '\n   ', ' current', '_', 'group', ' =', " ''", '\n   ', ' #', ' Initialize', ' a', ' variable', ' to', ' store', ' the', ' number', ' of', ' open', ' br', 'aces', '\n   ', ' open', '_', 'br', 'aces', ' =', ' ', '0', '\n   ', ' #', ' Loop', ' through', ' each', ' character', ' in', ' the', ' input', ' string', '\n   ', ' for', ' char', ' in', ' p', 'aren', '_', 'string', ':', '\n       ', ' #', ' If', ' the', ' character', ' is', ' an', ' open', ' br', 'ace', ',', ' increment', ' the', ' number', ' of', ' open', ' br', 'aces', '\n       ', ' if', ' char', ' ==', " '", "(':", '\n           ', ' open', '_', 'br', 'aces', ' +=', ' ', '1', '\n       ', ' #', ' If', ' the', ' character', ' is', ' a', ' closed', ' br', 'ace', ',', ' decrement', ' the', ' number', ' of', ' open', ' br', 'aces', '\n       ', ' elif', ' char', ' ==', " ')", "':", '\n           ', ' open', '_', 'br', 'aces', ' -=', ' ', '1', '\n       ', ' #', ' If', ' the', ' character', ' is', ' a', ' space', ',', ' ignore', ' it', '\n       ', ' elif', ' char', ' ==', " '", " ':", '\n           ', ' continue', '\n       ', ' #', ' If', ' the', ' number', ' of', ' open', ' br', 'aces', ' is', ' ', '0', ',', ' that', ' means', ' we', ' have', ' reached', ' the', ' end', ' of', ' a', ' group', '\n       ', ' if', ' open', '_', 'br', 'aces', ' ==', ' ', '0', ':', '\n           ', ' #', ' Append', ' the', ' current', ' group', ' to', ' the', ' list', ' of', ' groups', '\n           ', ' groups', '.', 'append', '(', 'current', '_', 'group', ')', '\n           ', ' #', ' Reset', ' the', ' current', ' group', ' and', ' the', ' number', ' of', ' open', ' br', 'aces', '\n           ', ' current', '_', 'group', ' =', " ''", '\n           ', ' open', '_', 'br', 'aces', ' =', ' ', '0', '\n       ', ' #', ' Otherwise', ',', ' add', ' the', ' character', ' to', ' the', ' current', ' group', '\n       ', ' else', ':', '\n           ', ' current', '_', 'group', ' +=', ' char', '\n   ', ' #', ' Return', ' the', ' list', ' of', ' groups', '\n   ', ' return', ' groups', '\n', '<|endoftext|>']
    # 31
    answer_list_31 = ['\n', 'def', ' is', '_', 'prime', '(', 'n', '):', '\n   ', ' """', 'Return', ' true', ' if', ' a', ' given', ' number', ' is', ' prime', ',', ' and', ' false', ' otherwise', '.', '\n   ', ' >>>', ' is', '_', 'prime', '(', '6', ')', '\n   ', ' False', '\n   ', ' >>>', ' is', '_', 'prime', '(', '1', '0', '1', ')', '\n   ', ' True', '\n   ', ' >>>', ' is', '_', 'prime', '(', '1', '1', ')', '\n   ', ' True', '\n   ', ' >>>', ' is', '_', 'prime', '(', '1', '3', '4', '4', '1', ')', '\n   ', ' True', '\n   ', ' >>>', ' is', '_', 'prime', '(', '6', '1', ')', '\n   ', ' True', '\n   ', ' >>>', ' is', '_', 'prime', '(', '4', ')', '\n   ', ' False', '\n   ', ' >>>', ' is', '_', 'prime', '(', '1', ')', '\n   ', ' False', '\n   ', ' """', '\n   ', ' if', ' n', ' ==', ' ', '1', ':', '\n       ', ' return', ' False', '\n   ', ' for', ' i', ' in', ' range', '(', '2', ',', ' n', '):', '\n       ', ' if', ' n', ' %', ' i', ' ==', ' ', '0', ':', '\n           ', ' return', ' False', '\n   ', ' return', ' True', '\n', '<|endoftext|>']
    # 35
    answer_list_35 = ['\n', 'def', ' max', '_', 'element', '(', 'l', ':', ' list', '):', '\n   ', ' """', 'Return', ' maximum', ' element', ' in', ' the', ' list', '.', '\n   ', ' >>>', ' max', '_', 'element', '([', '1', ',', ' ', '2', ',', ' ', '3', '])', '\n    ', '3', '\n   ', ' >>>', ' max', '_', 'element', '([', '5', ',', ' ', '3', ',', ' -', '5', ',', ' ', '2', ',', ' -', '3', ',', ' ', '3', ',', ' ', '9', ',', ' ', '0', ',', ' ', '1', '2', '3', ',', ' ', '1', ',', ' -', '1', '0', '])', '\n    ', '1', '2', '3', '\n   ', ' """', '\n   ', ' max', '_', 'element', ' =', ' l', '[', '0', ']', '\n   ', ' for', ' i', ' in', ' range', '(', '1', ',', ' len', '(', 'l', ')):', '\n       ', ' if', ' l', '[', 'i', ']', ' >', ' max', '_', 'element', ':', '\n           ', ' max', '_', 'element', ' =', ' l', '[', 'i', ']', '\n   ', ' return', ' max', '_', 'element', '\n', '<|endoftext|>']
    answer_list_array = [answer_list_0, answer_list_1, answer_list_31, answer_list_35]

    checkpoint = "bigcode/starcoder"
    num_layers = 40
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    all_layers = []
    target_0 = ''

    counter = 0
    for prompt, answer_list, useless_len in zip(prompt_array, answer_list_array, [119,119,119,119]):
        print(prompt)
        for target_token in answer_list:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            if counter >= useless_len:
                print(f"Generating token '{target_token}'")
                for layer in range(1, num_layers):
                    model.set_early_exit_layer(layer)
                    generated_ids = model.generate(
                        input_ids,
                        use_cache = True,
                        max_new_tokens=1,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    model.clear_early_exit_layer_indices()
                    last_ids = generated_ids[0][-1]
                    last_text = tokenizer.decode(last_ids)
                    print(f"layer {layer+1}: '{last_text}'")
                    if last_text == target_token:
                        # print(f"{target_token}: {layer+1}")
                        all_layers.append(layer+1)
            counter += 1
            prompt = prompt + target_token
        print(prompt)
        print(f"all_answers is {answer_list}")
        print(f"all_layers is {all_layers}")
        print(f"mean is {sum(all_layers) / len(all_layers)}")
        for i in range(len(answer_list)):
            print(f"{answer_list[i]}: {all_layers[i]}")
