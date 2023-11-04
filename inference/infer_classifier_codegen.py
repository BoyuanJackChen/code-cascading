from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch
from codegen_classifier import CodegenClassifier

prompt = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"

def main():
    tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/codegen-350M-mono", device_map="auto")
    load_start = time.time()
    model_original = AutoModelForCausalLM.from_pretrained(f"Salesforce/codegen-350M-mono", torch_dtype=torch.float32, device_map="auto")
    load_end = time.time()
    print(f"Model load time: {load_end - load_start}")
    model_original.eval()
    model = CodegenClassifier(model_original)
    print(f"Model copy time: {time.time() - load_end}")
    
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
    # input()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    loss, y_hat = model(input_ids)
    print(y_hat)

if __name__== "__main__":
    main()
