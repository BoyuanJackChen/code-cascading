import json

file_1 = "./answer/7B/7B_p0_l0.json"
file_2 = "./answer/7B/7B_original.json"
dict_1 = json.load(open(file_1, "r"))
dict_2 = json.load(open(file_2, "r"))

for i in range(0, len(dict_1)):
    if dict_1[i]["answer"] != dict_2[i]["answer"]:
        print(f"Answer {i} is different")