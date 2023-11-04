import json

with open('raw.json', 'r') as f:
    data = json.load(f)

pe_cpp = "<filename>solutions/solution_1.cpp\n// Here is the correct implementation of the code exercise in c++:\n"
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
    pe_index = answer.find(pe_cpp)
    answer = answer[pe_index+len(pe_cpp):]
    # trim answer
    if number_int in [27,32,41,92]:
        if "<|endoftext|>" in answer:
            answer = answer.split("<|endoftext|>")[0]
    else:
        answer = answer.split("main()")[0]
        last_newline_index = answer.rfind("\n")
        answer = answer[:last_newline_index]
    if number_int == 162:
        answer = answer.split("/*")[0]
    output_dict["answer"] = answer
    output_data.append(output_dict)

    print(f"Number {number}:")
    print(answer)
    if i%30==29:
        input()

with open('clean.json', 'w') as f:
    json.dump(output_data, f, indent=4)