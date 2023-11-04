import json

with open('16B_ee_w5_0.99.json', 'r') as f:
    data = json.load(f)

with open('../data/HumanEval.json', 'r') as f:
    original_data = json.load(f)

for j in range(len(data)):
    i = data[j]
    number = i["number"]
    number_int = int(number.split("/")[1])
    title = i["title"]
    answer = i["answer"]
    test = i["test"]
    if not answer.startswith("    "):
        print(number_int)
        print(answer)
        print(f"prompt is: ")
        print(original_data[j]["prompt"])
        input()
