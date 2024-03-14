from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

stop_words = [("\n", "\n"), "\n\n", "\nprint()"]
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
tokenizer.pad_token_id = tokenizer.eos_token_id
prompt = ["aga!", "aha!!!!!", "hohoho"]
input_ids = tokenizer.batch_encode_plus(prompt, padding=True, return_tensors="pt")["input_ids"]
print(input_ids)
# result = []
# for sw in stop_words:
#     print(f"stop word: {sw}; token id: {tokenizer.encode(sw)}")
#     result.append(tokenizer.encode(sw))
# print(result)
