import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('stats_wizard_he.csv')
model = '34B'

# Filter the DataFrame for the 7B model
df = df[df['model'] == model]
df["accuracy"] = df["accuracy"] * 100

# Plotting cost vs. accuracy
plt.figure(figsize=(10, 6))

# plt.scatter(df_7b['cost'], df_7b['accuracy'], color='blue')
for pass_at in df['pass_at'].unique():
    subset = df[df['pass_at'] == pass_at]
    plt.plot(subset['cost'], subset['accuracy'], marker='o', linestyle='-', label=f'Pick @: {pass_at}')

plt.title('Cost vs. Accuracy for 7B Model on HumanEval, pick@0,1,3,5,10, testlines=2,4', fontsize=14)
plt.xlabel('Cost ($)', fontsize=13)
plt.ylabel('Accuracy (%)', fontsize=13)
plt.legend()
plt.grid(True)
plt.show()
