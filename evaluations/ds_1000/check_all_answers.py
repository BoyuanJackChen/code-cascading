from ds1000 import DS1000Dataset
import json
import numpy as np

ds_data = DS1000Dataset("ds1000_data") # loads all questions into RAM

answer_file = "./34B_p10_t1_l0.json"
answer_data = json.load(open(answer_file, "r"))
all_correct = np.zeros(len(answer_data))
numpy_data = []
for answer in answer_data:
    if answer['library'] == "Numpy":
        numpy_data.append(answer)

for answer_dict in numpy_data:
    number = answer_dict["number"]
    answer = answer_dict["answer"]
    if answer.startswith(" "):
        answer = answer[1:]
    # print()
    # run official test on the generated code to 
    is_correct = ds_data["Numpy"][number].test(answer)
    print(number, is_correct)
    all_correct[number] = is_correct

print("Average correct:", np.mean(all_correct))