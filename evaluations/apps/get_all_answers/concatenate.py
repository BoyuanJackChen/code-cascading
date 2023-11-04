import json
from datasets import load_dataset

all_data = []
for start in range(0, 5000, 100):
    end = start + 99
    output_file_name = f"starcoder/start{start}_end{end}.json"
    with open(output_file_name, 'r') as f:
        data = json.load(f)
    all_data += data

result_file_name = "starcoder/test.json"
with open(result_file_name, 'w') as f:
    json.dump(all_data, f, indent=4)
