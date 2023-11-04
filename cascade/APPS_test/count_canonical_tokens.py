from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from datasets import load_dataset
from langdetect import detect
import json

# Load APPS dataset
all_questions_dict = load_dataset("codeparrot/apps", split="test")
number_key = "problem_id"
prompt_key = "question"
difficulty_key = "difficulty"
solutions_key = "solutions"
test_key = "input_output"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-16B-mono", device_map="auto")
all_solution_lengths = []
all_prompt_lengths = []
start = True

def is_english(text):
    try:
        language = detect(text)
        return language == 'en'
    except:
        return False


for question_dict in all_questions_dict:
    if start:
        for key in question_dict.keys():
            print(f"key is {key}")
            print(question_dict[key])
        start = False
    
    # Check if prompt is English
    prompt = question_dict[prompt_key]
    if not is_english(prompt):
        continue
    print(question_dict[number_key])
    
    # Find the shortest canonical solution
    canonical_collage = question_dict[solutions_key][2:-2]
    canonical_solution_list = canonical_collage.split('", "')
    canonical_idx = 0
    shortest_len = 100000
    for idx in range(len(canonical_solution_list)):
        if len(canonical_solution_list[idx]) < shortest_len:
            shortest_len = len(canonical_solution_list[idx])
            canonical_idx = idx
    canonical_solution = canonical_solution_list[canonical_idx]
    canonical_solution = canonical_solution.replace("\\n", "\n")
    prompt = question_dict[prompt_key]
    solution_ids = tokenizer(canonical_solution, return_tensors="pt").input_ids
    
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if len(prompt_ids) > 2048:
        print(f"\n{question_dict[number_key]}: {len(prompt_ids)} exceeds max of codegen.\n")
        input()
    all_solution_lengths.append(len(solution_ids[0]))
    all_prompt_lengths.append(len(prompt_ids[0]))

print(f"Number of questions: {len(all_solution_lengths)}")
print(f"Max of solutions: {max(all_solution_lengths)}; average: {sum(all_solution_lengths) / len(all_solution_lengths)}")
print(f"Max of prompts: {max(all_prompt_lengths)}; average: {sum(all_prompt_lengths)/len(all_prompt_lengths)}, sum: {sum(all_prompt_lengths)}")
# Get the percentage of all_solution_lengths smaller than 512
count = 0
for length in all_solution_lengths:
    if length < 512:
        count += 1
print(f"Percentage of all_solution_lengths smaller than 512: {count / len(all_solution_lengths)}")

# Make a histogram of all_lengths, with bins of length ranges
# Set up the figure and axis for the plot
fig, ax = plt.subplots()

# Create a histogram
ax.hist(all_solution_lengths, bins=range(0, max(all_solution_lengths)+10, 10), edgecolor='black')

# Set the title and labels
ax.set_title('Histogram of Tokenized Canonical Solution Lengths')
ax.set_xlabel('Length')
ax.set_ylabel('Number of Questions')

# Display the histogram
plt.tight_layout()
plt.show()

