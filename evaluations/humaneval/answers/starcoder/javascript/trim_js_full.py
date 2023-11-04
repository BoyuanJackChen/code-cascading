import json

with open('raw.json', 'r') as f:
    data = json.load(f)

pe_js = "<filename>solutions/solution_1.js\n// Here is the correct implementation of the code exercise in javascript:\n"
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
    
    pe_index = answer.find(pe_js)
    answer = answer[pe_index+len(pe_js):]

    output_dict["answer"] = answer
    output_data.append(output_dict)
    print(f"Number {number}:")
    print(answer)
    if i%30==29:
        input()

with open('full.json', 'w') as f:
    json.dump(output_data, f, indent=4)