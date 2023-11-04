import json

with open('16B_ee_w5_0.99.json', 'r') as f:
    data_1 = json.load(f)

with open('./codegen-mono/16B_k1.json', 'r') as f:
    data_2 = json.load(f)

for i in range(len(data_1)):
    dict_1 = data_1[i]
    dict_2 = data_2[i]
    number = dict_1["number"]
    number_int = int(number.split("/")[1])
    title = dict_1["title"]
    test = dict_1["test"]

    answer_1 = dict_1["answer"]
    answer_2 = dict_2["answer"]
    
    print(title, number_int)
    print(answer_1)
    print()
    print(answer_2)
    input()
