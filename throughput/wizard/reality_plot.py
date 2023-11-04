import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data
filename = 'wizard_he_1test.csv'
data = pd.read_csv(filename)

# Fill NaN values in 'Model' and 'Num GPUs Occupied' columns using forward fill method
data['Model'] = data['Model'].ffill()
data['Num GPUs Occupied'] = data['Num GPUs Occupied'].ffill()

# Convert k to integer and fill NaN with -1
data['k'].fillna(-1, inplace=True)
data['k'] = data['k'].astype(int)

all_combinations = [
    [5, 1, 0],
    [-1, 5, 0],
    [-1, 5, 10],
    [5, 1, 5],
    # [5, 3, 0],
    # [2, -1, 2]
]
all_expectations = [
    [0.002385, 62.16],
    [0.0333653, 64.25],
    [0.0622287, 65.01],
    [0.0081947, 62.33],
    # [0.0046325, 61.83],
    # [0.0169904, 60.42]
]
all_realities_raw = [
    [164.0, 34.1, 24.8, 58.89],
    [0.0, 164.0, 32.0, 61.27],
    [0.0, 164.0, 32.0, 63.75],
    [164.0, 34.1, 24.8, 60.0],
    # [164.0, 34.1, 21.9, 58.8],
    # [164.0, 0.0, 51.7, 56.9]
]
all_realities = []

# Iterate over each combination and its corresponding raw reality
for combination, raw_reality in zip(all_combinations, all_realities_raw):
    all_reality_entry = [0.0, raw_reality[-1]]
    # Calculate dollar consumption
    for k, model_num, model_size in zip(combination, [1, 2, 3], ["7B", "13B", "34B"]):
        if k == -1:
            continue
        # Filter dataframe for the specific model and k
        model_data = data[(data['Model'] == model_size) & (data['k'] == k)]
        cost_on_full_run = model_data['Cost on full run ($)'].values[0]
        t = raw_reality[model_num-1]/164 * cost_on_full_run
        all_reality_entry[0] = all_reality_entry[0] + t
    all_realities.append(all_reality_entry)

# Print all_realities
print(all_realities)

# ----- Plot -----
# Define a colormap
cmap = plt.cm.Blues
colors = {
    "7B": cmap(0.3),   # lighter blue
    "13B": cmap(0.5),  # medium blue
    "34B": cmap(0.7)   # darkest blue
}

# Set up the plot
plt.figure(figsize=(10, 6))
plt.title('Best Combinations & singular model pick@k for WizardCoder on HumanEval', size=18)
plt.xlabel('Cost on full run ($)', size=14)
plt.ylabel('Final Accuracy (%)', size=14)

# Plot data for each model
for name, group in data.groupby('Model'):
    plt.plot(group['Cost on full run ($)'], group['Final accu (%)'], marker='o', label=name, color=colors[name])

# Extract x and y values
ex_x = [point[0] for point in all_expectations]
ex_y = [point[1] for point in all_expectations]
real_x = [point[0] for point in all_realities]
real_y = [point[1] for point in all_realities]

# Plot the data
plt.scatter(ex_x, ex_y, c='blue', marker='x', label='Expectations')
for x, y, label in zip(ex_x, ex_y, all_combinations):
    annotation = f"({label[0]}, {label[1]}, {label[2]})"
    y_anno = 20 if (label == [5,1,5]) else 10
    plt.annotate(annotation, (x, y), textcoords="offset points", xytext=(0,y_anno), ha='center')
plt.scatter(real_x, real_y, c='red', marker='o', label='Realities')

# Display legend
plt.legend(title='Model')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()
