from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-1B-V1.0")
print(tokenizer.decode(0))
print(tokenizer.encode('</s>'))

filtered_b = [1,2,3,4,5,6]
stopping_ids = [[3,4,5], [2,4]]

def contains_subarray(filtered_b, subarray):
    sub_len = len(subarray)
    for i in range(len(filtered_b) - sub_len + 1):
        if filtered_b[i:i+sub_len] == subarray:
            return True
    return False
