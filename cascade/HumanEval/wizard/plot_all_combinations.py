import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame from the CSV data
df = pd.read_csv('stats_wizard_he.csv')
df['accuracy'] = df['accuracy'] * 100

# Plotting
plt.figure(figsize=(10, 6))
colors = ['mediumblue', 'darkblue', 'lightblue']
models = df['model'].unique()

for i, model in enumerate(models):
    model_data = df[df['model'] == model]
    plt.plot(model_data['cost'], model_data['accuracy'], label=model, color=colors[i], marker='o')

plt.xlabel('Cost ($)', fontsize=13)
plt.ylabel('Accuracy (%)', fontsize=13)
plt.title('WizardCoder-Python-V1.0 on HumanEval, pick@0,1,3,5,10, testlines=2,4', fontsize=15)
plt.legend()
plt.grid(True)
plt.show()