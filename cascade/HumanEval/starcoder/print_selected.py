import json

file_1 = "./selected/1B/1B_p0_t0_l0.json"
file_2 = "./selected/15B/15B_p0_t0_l0.json"

with open(file_1, "r") as f:
    data_1 = json.load(f)
with open(file_2, "r") as f:
    data_2 = json.load(f)

for dict_1, dict_2 in zip(data_1, data_2):
    number = dict_1["number"]
    correct_1 = dict_1["indeed"]
    correct_2 = dict_2["indeed"]
    if correct_1:
        continue
    answer_1 = dict_1["answer"]
    answer_2 = dict_2["answer"]
    print(f"--- Question {number}: Smaller model is incorrect, Larger model is {correct_2}. \n\n{answer_1}\n{answer_2}\n\n")
    input()
