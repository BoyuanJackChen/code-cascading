import json

file_0 = "./350M/350M_p0_t1_l0.json"
file_1 = "./350M/350M_p1_t1_l0.json"

all_0_data = []
with open(file_1, 'r') as f:
    all_1_data = json.load(f)

for i in range(len(all_1_data)):
    answer_dict = all_1_data[i]
    answer_dict["correct"] = False
    answer_dict["generated_testcode"] = ""
    all_0_data.append(answer_dict)

with open(file_0, 'w') as f:
    json.dump(all_0_data, f, indent=4)