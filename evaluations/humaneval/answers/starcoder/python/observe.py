import json

with open('clean.json', 'r') as f:
    answer_data = json.load(f)

with open('../../../data/HumanEval.json', 'r') as f:
    original_data = json.load(f)

for i in range(len(answer_data)):
    answer_dict = answer_data[i]
    prompt = original_data[i]["prompt"]
    answer = answer_dict["answer"]
    print(i)
    # print(prompt)
    print(answer)
    input()

with open('350M_k1.json', 'w') as f:
    json.dump(final_data, f, indent=4)