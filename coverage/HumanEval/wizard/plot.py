import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib_venn import venn3, venn3_circles, venn2, venn2_circles, venn2_unweighted
from venn import venn
import random

want_selected = False
random.seed(101)

# Load the CSV file
df = pd.read_csv('all_greedy_accuracies.csv')
title = "WizardCoder Family Correctness HumanEval full (164) questions correctness Venn Diagram"
if want_selected:
    selected_numbers = random.sample(range(1, 165), 49)
    df = df.loc[df['number'].isin(selected_numbers)]
    title = "WizardCoder HumanEval 30% (49) questions correctness Venn Diagram"

# Extract rows where each model answered correctly
correct_7B = set(df[df['7B'] == 1]['number'])
correct_13B = set(df[df['13B'] == 1]['number'])
correct_34B = set(df[df['34B'] == 1]['number'])
print(correct_7B)
print(correct_13B)
print(correct_34B)

# Calculate all subset sizes for the Venn diagram
AB = len(correct_7B.intersection(correct_13B)) - len(correct_7B.intersection(correct_13B, correct_34B))
AC = len(correct_7B.intersection(correct_34B)) - len(correct_7B.intersection(correct_13B, correct_34B))
BC = len(correct_13B.intersection(correct_34B)) - len(correct_7B.intersection(correct_13B, correct_34B))
A = len(correct_7B) - len(correct_7B.intersection(correct_13B)) - len(correct_7B.intersection(correct_34B)) + len(correct_7B.intersection(correct_13B, correct_34B))
B = len(correct_13B) - len(correct_7B.intersection(correct_13B)) - len(correct_13B.intersection(correct_34B)) + len(correct_7B.intersection(correct_13B, correct_34B))
C = len(correct_34B) - len(correct_7B.intersection(correct_34B)) - len(correct_13B.intersection(correct_34B)) + len(correct_7B.intersection(correct_13B, correct_34B))
ABC = len(correct_7B.intersection(correct_13B, correct_34B))

# Note: The values are in the following order:
# venn3 accepts subset sizes in the following order:
# (Abc, aBc, ABc, abC, AbC, aBC, ABC)
subset_sizes = (A, B, AB, C, AC, BC, ABC)

# Calculate the number of unsolved questions
total_questions = len(df)
unsolved = total_questions - sum(subset_sizes)

# Plot the Venn diagram
v = venn3(subsets=subset_sizes, set_labels=('7B', '13B', '34B'))
# Adjust the font size for set labels
for text in v.set_labels:
    text.set_fontsize(13)
for text in v.subset_labels:
    text.set_fontsize(11)


# Add annotations with total numbers
# These positions may need to be adjusted depending on your plot size and preferences
fontsize = 11
plt.text(-0.8, 0.3, f'Total: {len(correct_7B)}', fontsize=fontsize, color='red')
plt.text(0.65, 0.25, f'Total: {len(correct_13B)}', fontsize=fontsize, color='green')
plt.text(0.2, -0.7, f'Total: {len(correct_34B)}', fontsize=fontsize, color='blue')
plt.text(-0.9, -0.7, f'Unsolved: {unsolved}', fontsize=fontsize, color='black')
# plt.title(title)
plt.show()
