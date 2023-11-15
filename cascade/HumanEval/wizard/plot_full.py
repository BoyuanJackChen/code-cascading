import pandas as pd
import matplotlib.pyplot as plt

# Reading the CSV file
df = pd.read_csv('wizard_humaneval.csv')

# Data cleaning: Filling NaN values in 'k', converting 'k' to integers, and removing empty rows
df['k'] = df['k'].fillna(method='ffill').astype(int)  # Fill NaN values and convert to int
df = df.dropna(how='all')  # Remove rows where all elements are NaN
df = df[['k', 'Testlines', 'Cost on full run ($)', 'Overall Accuracy (%)']]  # Selecting necessary columns

# Plotting
plt.figure(figsize=(10, 6))

# Getting unique values of 'k'
k_values = df['k'].unique()

for k in k_values:
    subset = df[df['k'] == k]
    plt.plot(subset['Cost on full run ($)'], subset['Overall Accuracy (%)'], marker='o', label=f'k={k}')

plt.xlabel('Cost in $', fontsize=12)  # Increased font size for x-axis label
plt.ylabel('Accuracy in %', fontsize=12)  # Increased font size for y-axis label
plt.title('WizardCoder-LLAMA-7B on HumanEval, testlines=1,2,3,4', fontsize=14)  # Changed title with increased font size
plt.legend()
plt.grid(True)
plt.show()
