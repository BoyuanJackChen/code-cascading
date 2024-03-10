import json

# Load APPS Dataset
data_file = "../../../evaluations/apps/APPS_zeroshot_for_code_generation_intro_prompt_only.jsonl"
all_questions_dict = []
with open(data_file, "r") as f:
    for line in f:
        json_dict = json.loads(line)
        prompt = json_dict["prompt"]
        # prompt = prompt.split("\n")
        # prompt = "\n".join(prompt[1:-4])
        print(prompt)
        print("A")
        input()