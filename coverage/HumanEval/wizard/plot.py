import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib_venn import venn3, venn3_circles, venn2, venn2_circles, venn2_unweighted
from venn import venn
import random

want_selected = False

# Load the CSV file
title = "WizardCoder Family Correctness HumanEval full (164) questions correctness Venn Diagram"
if want_selected:
    random.seed(101)
    selected_numbers = random.sample(range(1, 165), 49)
    title = "WizardCoder HumanEval 30% (49) questions correctness Venn Diagram"

# Extract rows where each model answered correctly
correct_7B  = [0, 1, 2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 38, 40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 66, 68, 69, 71, 72, 78, 80, 81, 82, 83, 85, 86, 88, 92, 94, 96, 97, 98, 102, 104, 105, 106, 107, 112, 116, 117, 121, 122, 124, 128, 133, 136, 143, 146, 148, 149, 152, 153, 154, 155, 156, 158, 159, 160, 161, 162]
correct_13B = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 76, 79, 82, 85, 86, 87, 89, 92, 94, 95, 96, 97, 98, 103, 105, 107, 112, 114, 116, 121, 123, 124, 125, 128, 133, 136, 141, 142, 143, 144, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 161, 162]
correct_34B = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 69, 70, 71, 72, 73, 74, 78, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 106, 107, 112, 114, 116, 117, 122, 123, 126, 128, 133, 136, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162]
correct_7B = set(correct_7B)
correct_13B = set(correct_13B)
correct_34B = set(correct_34B)
# Find numbers that are in 13B but not 7B and 34B
print(correct_13B - correct_7B - correct_34B)
input()

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
total_questions = 164
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
fontsize = 11.5
plt.text(-0.85, 0.3, f'Total: {len(correct_7B)}', fontsize=fontsize, color='red')
plt.text(0.65, 0.25, f'Total: {len(correct_13B)}', fontsize=fontsize, color='green')
plt.text(0.3, -0.7, f'Total: {len(correct_34B)}', fontsize=fontsize, color='blue')
plt.text(-0.85, -0.7, f'Unsolvable: {unsolved}', fontsize=fontsize, color='black')
# plt.title(title)
plt.show()
