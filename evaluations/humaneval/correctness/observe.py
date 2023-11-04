import json

# Load HumanEval_difficulty.json
with open("HumanEval_difficulty.json", "r") as f:
    data = json.load(f)

wanted_difficulty = 2
for prompt_dict in data:
    difficulty = prompt_dict["difficulty"]
    if difficulty != wanted_difficulty:
        continue
    prompt = prompt_dict["prompt"]
    number = prompt_dict["number"]
    print(number)
    print(prompt)
    input()