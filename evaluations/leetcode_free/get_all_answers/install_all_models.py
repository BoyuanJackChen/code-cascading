from transformers import AutoTokenizer, AutoModelForCausalLM
import os

if __name__== "__main__":
    os.environ['TRANSFORMERS_CACHE'] = '/vast/bc3194/huggingface_cache'
    model = AutoModelForCausalLM.from_pretrained(f"Salesforce/codegen-350M-multi", device_map="auto")
    print("350M loaded")
    model = AutoModelForCausalLM.from_pretrained(f"Salesforce/codegen-2B-multi", device_map="auto")
    print("2B loaded")
    model = AutoModelForCausalLM.from_pretrained(f"Salesforce/codegen-6B-multi", device_map="auto")
    print("6B loaded")
    model = AutoModelForCausalLM.from_pretrained(f"Salesforce/codegen-16B-multi", device_map="auto")
    print("16B loaded")