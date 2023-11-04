import json

# Load py_top_p.json
py_top_p = json.load(open("py_top_p.json", "r"))

for data_dict in py_top_p:
    answer = data_dict["answer"]
    number = data_dict["number"]
    number_int = int(number.split('/')[1])
    print(answer)
    input()