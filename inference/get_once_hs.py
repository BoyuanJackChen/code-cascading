from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CodeGenModel
import torch

eos_token = 50256
stop_words = []
# 0  (medium)
prompt = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"

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
    checkpoint = "Salesforce/codegen-16B-mono"
    num_layers = 20
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    print("initialized")
    # 0 16B
    answer_list = ['    ', 'for', ' i', ' in', ' range', '(', 'len', '(', 'n', 'umbers', ')', '):', '\n', '        ','for', ' j', ' in', ' range', '(', 'i', ' +', ' 1', ',', ' len', '(', 'n', 'umbers', ')', '):', '\n','            ', 'if', ' abs', '(', 'n', 'umbers', '[', 'i', ']', ' -', ' numbers', '[', 'j', '])',' <', ' threshold', ':', '\n', '                ', 'return', ' True', '\n', '    ', 'return',' False', '\n\n']
    answer_tokens = [tokenizer.encode(answer, add_special_tokens=False)[0] for answer in answer_list]
    model.set_oracle(answer_tokens)

    print(f"\n{prompt}\n")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(
        input_ids,
        eos_token_id=eos_token,
        pad_token_id=eos_token,
        use_cache = False,
        max_new_tokens=5,
        num_beam_groups=4,
        diversity_penalty=0.3,
        num_beams=4,
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
    all_layers = model.get_early_exit_layer_indices()
    all_layers = all_layers[:all_layers.count_nonzero()].type(torch.int64).tolist()

    print(f"all_answers is {answer_list}")
    print(f"all_layers is {all_layers}")
    print(f"mean is {sum(all_layers) / len(all_layers)}")
    for i in range(len(all_layers)):
        print(f"{answer_list[i]}: {all_layers[i]}")
