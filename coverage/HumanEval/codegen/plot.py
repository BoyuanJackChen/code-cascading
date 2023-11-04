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
title = "Codegen HumanEval full (164) questions correctness Venn Diagram"
if want_selected:
    selected_numbers = random.sample(range(1, 165), 49)
    df = df.loc[df['number'].isin(selected_numbers)]
    title = "Codegen HumanEval 30% (49) questions correctness Venn Diagram"

# Extract rows where each model answered correctly
correct_350M = set(df[df['350M'] == 1]['number'])
correct_2B = set(df[df['2B'] == 1]['number'])
correct_6B = set(df[df['6B'] == 1]['number'])
correct_16B = set(df[df['16B'] == 1]['number'])
print(correct_350M)
print(correct_2B)
print(correct_6B)
print(correct_16B)

# Calculate all subset sizes for the Venn diagram
AB = len(correct_2B.intersection(correct_6B)) - len(correct_2B.intersection(correct_6B, correct_16B))
AC = len(correct_2B.intersection(correct_16B)) - len(correct_2B.intersection(correct_6B, correct_16B))
BC = len(correct_6B.intersection(correct_16B)) - len(correct_2B.intersection(correct_6B, correct_16B))
A = len(correct_2B) - len(correct_2B.intersection(correct_6B)) - len(correct_2B.intersection(correct_16B)) + len(correct_2B.intersection(correct_6B, correct_16B))
B = len(correct_6B) - len(correct_2B.intersection(correct_6B)) - len(correct_6B.intersection(correct_16B)) + len(correct_2B.intersection(correct_6B, correct_16B))
C = len(correct_16B) - len(correct_2B.intersection(correct_16B)) - len(correct_6B.intersection(correct_16B)) + len(correct_2B.intersection(correct_6B, correct_16B))
ABC = len(correct_2B.intersection(correct_6B, correct_16B))

# Note: The values are in the following order:
# venn3 accepts subset sizes in the following order:
# (Abc, aBc, ABc, abC, AbC, aBC, ABC)
subset_sizes = (A, B, AB, C, AC, BC, ABC)

# Plot the Venn diagram
venn3(subsets=subset_sizes, set_labels=('2B', '6B', '16B'))
plt.title(title)

# Add annotations with total numbers
# These positions may need to be adjusted depending on your plot size and preferences
plt.text(-0.8, 0.3, f'Total: {len(correct_2B)}', fontsize=10, color='red')
plt.text(0.7, 0.2, f'Total: {len(correct_6B)}', fontsize=10, color='green')
plt.text(0.2, -0.7, f'Total: {len(correct_16B)}', fontsize=10, color='blue')

plt.show()