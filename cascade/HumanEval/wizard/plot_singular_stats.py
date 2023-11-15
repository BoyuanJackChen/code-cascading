import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('stats_wizard_he.csv')

# Filter the DataFrame for the 7B model
df_7b = df[df['model'] == '7B']

# Plotting cost vs. accuracy
plt.figure(figsize=(10, 6))
plt.scatter(df_7b['cost'], df_7b['accuracy'], color='blue')
plt.title('Cost vs. Accuracy for 7B Model')
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
