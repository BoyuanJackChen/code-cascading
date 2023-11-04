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

    # Trim end of text
    if answer.endswith("<|endoftext|>"):
        answer = answer[:-len("<|endoftext|>")]
    
    # Remove all lines that start with // or console.log:
    lines = answer.split("\n")
    new_lines = []
    for line in lines:
        if not line.startswith("//") and not line.startswith("console.log"):
            new_lines.append(line)
    answer = "\n".join(new_lines)

    # Trim extra function completions
    extra_start = "// Here is the correct implementation of the code"
    if extra_start in answer:
        extra_start_index = answer.find(extra_start)-2
        answer = answer[:extra_start_index]

    # Trim at the end of a function
    end_start_index = answer.find("};\n\n")
    if end_start_index != -1:
        answer = answer[:end_start_index+3]
    else:
        comment_start_index = answer.find("}\n\n")
        if comment_start_index != -1:
            answer = answer[:comment_start_index+3]

    # Remove empty lines in answer
    lines = answer.split("\n")
    new_lines = []
    for line in lines:
        if line != "":
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