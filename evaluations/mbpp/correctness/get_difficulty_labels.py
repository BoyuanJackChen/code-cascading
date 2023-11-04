import pandas as pd
import json

correctness_file = "codegen-mono/mbpp-mono-k1.csv"
data_file = "../sanitized-mbpp.jsonl"
output_data_file = "mbpp_d.json"

number_key = "task_id"
prompt_key = "prompt"

def get_difficulty(small, big):
    if small==1:
        return 0
    elif big==1:
        return 1
    else:
        return 2

# Load data_file to data with json
with open(data_file) as f:
    data = json.load(f)
print(len(data))

# load humaneval_mono_k1.csv
df = pd.read_csv(correctness_file)
df = df[["350M", "16B"]]

result_list = []
for i in range(len(data)):
    data_dict = data[i]
    number = f"MBPP/{data_dict[number_key]}"
    output_dict = dict()
    difficulty = get_difficulty(df.iloc[i]["350M"], df.iloc[i]["16B"])
    output_dict["number"] = number
    output_dict["prompt"] = data_dict[prompt_key]
    output_dict["difficulty"] = difficulty
    result_list.append(output_dict)

with open(output_data_file, "w") as f:
    json.dump(result_list, f, indent=4)