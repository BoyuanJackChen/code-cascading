import json
from datasets import load_dataset

dir = "0000/"

# load a json file
with open(dir+'question.txt', 'r') as f:
    question = f.read()
with open(dir+'input_output.json', 'r') as f:
    input_output = json.load(f)
with open(dir+'solutions.json', 'r') as f:
    solutions = json.load(f)
with open(dir+'metadata.json', 'r') as f:
    metadata = json.load(f)

# question remove all new line tokens
question = question.replace('\n\n', '\n')
print(f"Question:\n{question}\n")
print(f"Input Output:\n{input_output}\n")
print(f"Solutions:\n{solutions[0]}\n")
print(f"Metadata:\n{metadata}\n")

# Load APPS dataset
all_questions_dict = load_dataset("codeparrot/apps", split="test")
for question_dict in all_questions_dict:
    print(question_dict)
    print(question_dict.keys())
    print(question_dict['problem_id'])
    input()