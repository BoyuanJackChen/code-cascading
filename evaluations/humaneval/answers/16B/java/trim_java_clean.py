import json

with open('raw.json', 'r') as f:
    data = json.load(f)

output_data = []

for i in range(len(data)):
    question_dict = data[i]
    output_dict = question_dict
    number = question_dict["number"]
    number_int = int(number.split("/")[1])
    prompt = question_dict['prompt']
    answer = question_dict["answer"]
    test = question_dict["test"]
    text = question_dict["text"]
    example_test = question_dict["example_test"]
    declaration = question_dict["declaration"]

    if answer.startswith(prompt):
        answer = answer[len(prompt):]
    else:
        # Remove all lines that starts with //
        lines = answer.split("*/\n")
        new_lines = []
        for j in range(1, len(lines)):
            new_lines.append(lines[j])
        answer = "\n".join(new_lines)
        lines = answer.split("\n")
        new_lines = []
        for k in range(1, len(lines)):
            new_lines.append(lines[k])
        answer = "\n".join(new_lines)

    # Trim end of text
    if answer.endswith("<|endoftext|>"):
        answer = answer[:-len("<|endoftext|>")]
    
    # Trim at the first occurrence of '''
    if "'''" in answer:
        answer = answer[:answer.find("'''")]
    
    # Trim at the first occurrence of \n\n
    if "\n\n" in answer:
        answer = answer[:answer.find("\n\n")]

    # Remove all lines that starts with //
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