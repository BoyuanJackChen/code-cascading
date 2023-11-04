import json

with open('raw.json', 'r') as f:
    data = json.load(f)

pe_go = "<filename>solutions/solution_1.go\n// Here is the correct implementation of the code exercise in go:\n"
output_data = []

for i in range(len(data)):
    question_dict = data[i]
    output_dict = question_dict
    number = question_dict["number"]
    number_int = int(number.split("/")[1])
    prompt = question_dict['prompt']
    answer = question_dict["answer"]
    test = question_dict["test"]
    test_setup = question_dict["test_setup"]
    example_test = question_dict["example_test"]
    declaration = question_dict["declaration"]
    docstring = question_dict["docstring"]

    pe_index = answer.find(pe_go)
    answer = answer[pe_index+len(pe_go):]

    # Cut end of text
    if answer.endswith("<|endoftext|>"):
        answer = answer[:-len("<|endoftext|>")]

    # Trim at the start of the function
    lines = prompt.split("\n")
    last_func_line = None
    for line in reversed(lines):
        if line.startswith("func"):
            last_func_line = line
            break
    func_start = last_func_line.split("(")[0]
    if func_start in answer:
        func_start_index = answer.find(func_start)
        answer = answer[func_start_index:]
    else:
        func_start_index = answer.find("func")
        answer = answer[func_start_index:]

    # Get the last occurrence of "\n}\n\n"
    end_keyword = "\n}\n\n"
    if end_keyword in answer:
        last_occurrence = answer.find(end_keyword) + 4
        answer = answer[:last_occurrence]

    # Remove all lines that starts with "//" in answer
    lines = answer.split("\n")
    new_lines = []
    for line in lines:
        if not line.startswith("//"):
            new_lines.append(line)
    answer = "\n".join(new_lines)

    output_dict["answer"] = answer
    output_data.append(output_dict)

    print(f"Number {number}:")
    print(answer)
    if i%30==29:
        input()

with open('clean.json', 'w') as f:
    json.dump(output_data, f, indent=4)