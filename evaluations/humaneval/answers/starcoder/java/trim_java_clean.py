import json

with open('raw.json', 'r') as f:
    data = json.load(f)

pe_jv = "<filename>solutions/solution_1.java\n// Here is the correct implementation of the code exercise in java:\n"
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

    pe_index = answer.find(pe_jv)
    answer = answer[pe_index+len(pe_jv):]

    # Trim end of text
    if answer.endswith("<|endoftext|>"):
        answer = answer[:-len("<|endoftext|>")]
    
    # Trim extra function completions
    extra_start = "// Here is the correct implementation of the code"
    if extra_start in answer:
        extra_start_index = answer.find(extra_start)-3
        answer = answer[:extra_start_index]

    # # Trim at the start of the function
    # lines = prompt.split("\n")
    # last_func_line = None
    # for line in reversed(lines):
    #     if line.startswith("\tpublic") or line.startswith("    public"):
    #         last_func_line = line
    #         break
    # if last_func_line is not None:
    #     func_start = last_func_line.split("(")[0]
    #     if func_start in answer:
    #         func_start_index = answer.find(func_start)
    #         answer = answer[func_start_index:]
    #     else:
    #         func_start_index = answer.find("public")
    #         answer = answer[func_start_index:]

    # # Trim off the last extra '}' for class
    # if answer.endswith("}\n}\n") or answer.endswith("}\n}\n\n"):
    #     answer = answer[:-3]

    output_dict["answer"] = answer
    output_data.append(output_dict)

    print(f"Number {number}:")
    print(answer)
    if i%30==29:
        input()

with open('clean.json', 'w') as f:
    json.dump(output_data, f, indent=4)