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
    
    # Remove all lines that start with # or print:
    lines = answer.split("\n")
    new_lines = []
    for line in lines:
        if not line.startswith("#") and not line.startswith("print") and not line.startswith("    #"):
            new_lines.append(line)
    answer = "\n".join(new_lines)

    # Trim after triple new line
    end_start_index = answer.find('\n\n\n')
    if end_start_index != -1 and number_int not in [85, 162]:
        answer = answer[:end_start_index+1]

    # Trim after double new line, followed by anything that's not an empty space.
    match = re.search(r'\n\n[^ ]', answer)
    if match and number_int not in [85, 162]:
        end_start_index = match.start()
        answer = answer[:end_start_index+1]
    
    # Trim off the first new line if it exists
    if answer.startswith("\n"):
        answer = answer[1:]

    output_dict["answer"] = answer
    output_data.append(output_dict)
    print(f"Number {number}:")
    print(answer)
    if i%30==29:
        input()

with open('clean.json', 'w') as f:
    json.dump(output_data, f, indent=4)