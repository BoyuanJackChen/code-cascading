import json

with open('350M_k1.jsonl', 'r') as f:
    # Read in each line as a dictionary
    original_data = [json.loads(line) for line in f]

with open('../../data/HumanEval.json', 'r') as f:
    humaneval = json.load(f)

final_data = []
for i in range(len(original_data)):
    data_dict = original_data[i]
    data_dict["number"] = data_dict.pop("task_id")
    data_dict["title"] = humaneval[i]["entry_point"]
    data_dict["answer"] = data_dict.pop("completion")
    data_dict["test"] = humaneval[i]["test"]
    final_data.append(data_dict)

with open('350M_k1.json', 'w') as f:
    json.dump(final_data, f, indent=4)