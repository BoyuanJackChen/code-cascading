from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import json

# Load HumanEval Dataset
all_questions_dict = json.load(open("../../evaluations/mbpp/mbpp.json", "r"))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-16B-mono", device_map="auto")
all_lengths = []
all_prompt_lengths = []

for question_dict in all_questions_dict:
    canonical_solution = question_dict["code"]
    canonical_solution = canonical_solution.replace(" \r", "")
    canonical_solution = canonical_solution.replace("\r", "")
    prompt = question_dict["text"]
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    solution_ids = tokenizer(canonical_solution, return_tensors="pt").input_ids
    all_lengths.append(len(solution_ids[0]))
    all_prompt_lengths.append(len(prompt_ids[0]))
    
print(sum(all_prompt_lengths))
print(sum(all_lengths))

# print the index with the highest value
print(all_lengths.index(max(all_lengths)))

# Make a histogram of all_lengths, with bins of length ranges
# Set up the figure and axis for the plot
fig, ax = plt.subplots()

# Create a histogram
ax.hist(all_lengths, bins=range(0, max(all_lengths) + 10, 10), edgecolor='black')

# Set the title and labels
ax.set_title('Histogram of Tokenized Canonical Solution Lengths')
ax.set_xlabel('Length')
ax.set_ylabel('Number of Questions')

# Display the histogram
plt.tight_layout()
plt.show()
