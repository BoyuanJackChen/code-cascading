from datasets import load_dataset

all_questions_dict = load_dataset("codeparrot/apps", split="test")
print(all_questions_dict)

for i in all_questions_dict:
    print(i)
    print(i.keys())
    print(i["question"])
    input()