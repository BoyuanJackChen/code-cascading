import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from langdetect import detect
import pandas as pd
import matplotlib.pyplot as plt

def is_english(text):
    try:
        language = detect(text)
    except:
        return False
    if language == 'en':
        return True
    else:
        return False

checkpoint = "Salesforce/codegen-16B-mono"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")

# Create a pandas dataframe of 2 columns: task_id and len_ids
df = pd.DataFrame(columns=['problem_id', 'len_ids'])

# all_questions_dict = load_dataset("codeparrot/apps", split="test")
# for question_dict in all_questions_dict:
#     prompt = question_dict['question']
#     # detect if it is english
#     if not is_english(prompt):
#         continue
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
#     prompt_len = len(input_ids[0])
#     task_id = question_dict['problem_id']
#     df = df.append({'problem_id': task_id, 'len_ids': prompt_len}, ignore_index=True)
# df.to_csv('len_stats.csv', index=False)

df = pd.read_csv('len_stats.csv')
print(len(df))
print(df[df['len_ids'] > 2048].count())
print(df[df['len_ids'] > 1500].count())
