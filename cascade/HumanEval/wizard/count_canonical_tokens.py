from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import json

# Load HumanEval Dataset
all_questions_dict = json.load(open("../../../evaluations/humaneval/data/HumanEval_py.json", "r"))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-3B-V1.0", device_map="auto")
all_lengths = []

first = True
for question_dict in all_questions_dict:
    canonical_solution = question_dict["canonical_solution"]
    solution_ids = tokenizer(canonical_solution, return_tensors="pt").input_ids
    if first:
        print(solution_ids)
        first = False
    all_lengths.append(len(solution_ids[0]))

# print the max of all_lengths
print(max(all_lengths))

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

