import json
import re

with open('raw.json', 'r') as f:
    data = json.load(f)

pe_py = "<filename>solutions/solution_1.py\n// Here is the correct implementation of the code exercise in python:\n"
output_data = []

for i in range(len(data)):
    question_dict = data[i]
    output_dict = question_dict
    number = question_dict["number"]
    number_int = int(number.split("/")[1])
    prompt = question_dict['prompt']
    answer = question_dict["answer"]
    test = question_dict["test"]
    example_test = question_dict["example_test"]
    declaration = question_dict["declaration"]

    pe_index = answer.find(pe_py)
    answer = answer[pe_index+len(pe_py):]

    # Trim end of text
    if answer.endswith("<|endoftext|>"):
        answer = answer[:-len("<|endoftext|>")]
    
    # Trim extra function completions
    extra_start = "// Here is the correct implementation of the code"
    if extra_start in answer:
        extra_start_index = answer.find(extra_start)-2
        answer = answer[:extra_start_index]

    output_dict["answer"] = answer
    output_data.append(output_dict)
    print(f"Number {number}:")
    print(answer)
    if i%30==29:
        input()

with open('clean.json', 'w') as f:
    json.dump(output_data, f, indent=4)