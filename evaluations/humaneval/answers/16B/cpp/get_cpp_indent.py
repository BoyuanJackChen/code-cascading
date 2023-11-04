import json

with open('full.json', 'r') as f:
    data = json.load(f)

with open('../../../data/HumanEval_cpp_trimmed.json', 'r') as f:
    original_data = json.load(f)

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

    # replace all "    " with "\t" in answer
    prompt = prompt.replace("            ", "\t\t\t")
    prompt = prompt.replace("        ", "\t\t")
    prompt = prompt.replace("    ", "\t")
    original_prompt = original_data[i]["prompt"]
    if prompt != original_prompt:
        print(f"Number {number} has different prompt!")
        print(prompt)
        print(original_prompt)
        input()

    output_dict["answer"] = answer
    output_data.append(output_dict)

    print(f"Number {number}:")
    print(answer)
    if i%30==29:
        input()

with open('clean.json', 'w') as f:
    json.dump(output_data, f, indent=4)