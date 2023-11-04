import json

language = "py"

# # Convert jsonl to json
# with open(f'HumanEval-{language}.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f.readlines()]
# with open(f'HumanEval_{language}.json', 'w') as f:
#     json.dump(data, f, indent=4)

with open(f'HumanEval_{language}.json', 'r') as f:
    data = json.load(f)

for i in data:
    prompt = i["prompt"]
    test_text = i["test"]
    task_id = i["task_id"]
    # test_text = test_text.split("\n\n\n")[1]
    # print(test_text)
    print(prompt)
    print(task_id)
    input()