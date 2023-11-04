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
    answer = answer[len(prompt):]
    if answer.endswith("<|endoftext|>"):
        answer = answer[:-len("<|endoftext|>")]
    
    if i in [117, 141]:
        print(f"Number {number}:")
        print(prompt)
        print()
        print(answer)
        input()

with open('clean.json', 'w') as f:
    json.dump(output_data, f, indent=4)