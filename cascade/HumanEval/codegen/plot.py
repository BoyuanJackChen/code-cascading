import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplcursors

# Load all_combinations_test.csv
input_file = "./all_comb/5_test.csv"
df = pd.read_csv(input_file).fillna(method='ffill')
df[['k1', 'k2', 'k3', 'k4']] = df[['k1', 'k2', 'k3', 'k4']].astype(int)
fig, ax = plt.subplots()

selected_dots = [
    [-1.0, -1.0, -1.0, 10.0],
    [-1.0, -1.0, 2.0, 0.0],
    [-1.0, -1.0, 2.0, 4.0],
    [-1.0, -1.0, 5.0, 10.0],
    [-1.0, 0.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0, 0.0],
    [-1.0, 1.0, 2.0, -1.0],
    [-1.0, 1.0, 2.0, 0.0],
    [-1.0, 1.0, 2.0, 4.0],
    [-1.0, 1.0, 5.0, 0.0],
    [-1.0, 1.0, 5.0, 10.0],
    [-1.0, 4.0, -1.0, -1.0],
    [0.0, -1.0, -1.0, -1.0],
    [1.0, 0.0, -1.0, -1.0],
    [2.0, 0.0, -1.0, -1.0],
]
baseline_dots = [
    [0, -1, -1, -1],
    [-1, 1, 1, -1],
    [-1, -1, 0, -1],
    [-1, -1, 2, -1],
    [-1, -1, -1, 0],
    [-1, -1, -1, 4],
    [-1, -1, -1, 10],
]
model_1 = '350M'
model_2 = '2B'
model_3 = '6B'
model_4 = '16B'
alpha_value = 0.5
lined_alpha_value = 0.3

# Find Pareto points
if len(selected_dots) == 0:
    selected_df = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        cost_condition = df['cost'] < row['cost']
        accuracy_condition = df['accuracy'] >= row['accuracy']
        if not any(cost_condition & accuracy_condition):
            selected_df = pd.concat([selected_df, pd.DataFrame([row.values], columns=df.columns)])
    print_array = selected_df[['k1', 'k2', 'k3', 'k4']].to_numpy()
    # Print this array in python format, separated with comma
    for row in print_array:
        print(f"    [{row[0]}, {row[1]}, {row[2]}, {row[3]}],")
else:
    selected_rows = []
    for dot in selected_dots:
        matching_row = df[(df['k1'] == dot[0]) & (df['k2'] == dot[1]) & (df['k3'] == dot[2]) & (df['k4'] == dot[3])]
        selected_rows.append(matching_row)
    selected_df = pd.concat(selected_rows)
    
# Define the color for the line and the dots on the line
colors = ['lightskyblue', 'dodgerblue', 'royalblue', 'mediumblue']
dots_color = 'lightblue'

# Plotting primary scatter plot first
scatter = ax.scatter(df["cost"], df["accuracy"], c=dots_color, marker='o', label='Each k1, k2, k3, k4 combination')

# Connecting the specific dots with a line
combinations_1 = [[0, -1, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1], [3, -1, -1, -1], [4, -1, -1, -1], [5, -1, -1, -1], [10, -1, -1, -1]]
combinations_2 = [[-1, 0, -1, -1], [-1, 1, -1, -1], [-1, 2, -1, -1], [-1, 3, -1, -1], [-1, 4, -1, -1], [-1, 5, -1, -1], [-1, 10, -1, -1]]
combinations_3 = [[-1, -1, 0, -1], [-1, -1, 1, -1], [-1, -1, 2, -1], [-1, -1, 3, -1], [-1, -1, 4, -1], [-1, -1, 5, -1], [-1, -1, 10, -1]]
combinations_4 = [[-1, -1, -1, 0], [-1, -1, -1, 1], [-1, -1, -1, 2], [-1, -1, -1, 3], [-1, -1, -1, 4], [-1, -1, -1, 5], [-1, -1, -1, 10]]

# Create an empty set with columns same as df
subset_1 = pd.DataFrame(columns=df.columns)
subset_2 = pd.DataFrame(columns=df.columns)
subset_3 = pd.DataFrame(columns=df.columns)
subset_4 = pd.DataFrame(columns=df.columns)

# Plot combinations_1
x_vals_1, y_vals_1 = [], []
for combo in combinations_1:
    subset = df[(df['k1'] == combo[0]) & (df['k2'] == combo[1]) & (df['k3'] == combo[2]) & (df['k4'] == combo[3])]
    if not subset.empty:
        x_vals_1.append(subset["cost"].values[0])
        y_vals_1.append(subset["accuracy"].values[0])
        subset_1 = pd.concat([subset_1, subset])
    else:
        print(f"Combination {combo} not found in the dataframe.")
if x_vals_1 and y_vals_1:
    ax.plot(x_vals_1, y_vals_1, c=colors[0], label=model_1, alpha=alpha_value)  # Specify color and label for combinations_1
    ax.scatter(x_vals_1, y_vals_1, c=colors[0], marker='o', alpha=lined_alpha_value)

# Plot combinations_2
x_vals_2, y_vals_2 = [], []
for combo in combinations_2:
    subset = df[(df['k1'] == combo[0]) & (df['k2'] == combo[1]) & (df['k3'] == combo[2] & (df['k4'] == combo[3]))]
    if not subset.empty:
        x_vals_2.append(subset["cost"].values[0])
        y_vals_2.append(subset["accuracy"].values[0])
        subset_2 = pd.concat([subset_2, subset])
    else:
        print(f"Combination {combo} not found in the dataframe.")
if x_vals_2 and y_vals_2:  
    ax.plot(x_vals_2, y_vals_2, c=colors[1], label=model_2, alpha=alpha_value)  # Specify color and label for combinations_2
    ax.scatter(x_vals_2, y_vals_2, c=colors[1], marker='o', alpha=lined_alpha_value)

# Plot combinations_3
x_vals_3, y_vals_3 = [], []
for combo in combinations_3:
    subset = df[(df['k1'] == combo[0]) & (df['k2'] == combo[1]) & (df['k3'] == combo[2]) & (df['k4'] == combo[3])]
    if not subset.empty:
        x_vals_3.append(subset["cost"].values[0])
        y_vals_3.append(subset["accuracy"].values[0])
        subset_3 = pd.concat([subset_3, subset])
    else:
        print(f"Combination {combo} not found in the dataframe.")
if x_vals_3 and y_vals_3:  
    ax.plot(x_vals_3, y_vals_3, c=colors[2], label=model_3, alpha=alpha_value)  # Specify color and label for combinations_3
    ax.scatter(x_vals_3, y_vals_3, c=colors[2], marker='o', alpha=lined_alpha_value)

# Plot combinations_4
x_vals_4, y_vals_4 = [], []
for combo in combinations_4:
    subset = df[(df['k1'] == combo[0]) & (df['k2'] == combo[1]) & (df['k3'] == combo[2]) & (df['k4'] == combo[3])]
    if not subset.empty:
        x_vals_4.append(subset["cost"].values[0])
        y_vals_4.append(subset["accuracy"].values[0])
        subset_4 = pd.concat([subset_4, subset])
    else:
        print(f"Combination {combo} not found in the dataframe.")
if x_vals_4 and y_vals_4:
    ax.plot(x_vals_4, y_vals_4, c=colors[3], label=model_4, alpha=alpha_value)  # Specify color and label for combinations_4
    ax.scatter(x_vals_4, y_vals_4, c=colors[3], marker='o', alpha=lined_alpha_value)

# Find and plot the best baseline
if len(baseline_dots) == 0:
    subset_all = pd.concat([subset_1, subset_2, subset_3, subset_4])
    print(subset_all)
else:
    baseline_rows = []
    for dot in baseline_dots:
        matching_row = df[(df['k1'] == dot[0]) & (df['k2'] == dot[1]) & (df['k3'] == dot[2]) & (df['k4'] == dot[3])]
        baseline_rows.append(matching_row)
    subset_all = pd.concat(baseline_rows)
# Get the pareto points in subset_all
baseline_pareto = pd.DataFrame(columns=subset_all.columns)
for index, row in subset_all.iterrows():
    cost_condition = subset_all['cost'] < row['cost']
    accuracy_condition = subset_all['accuracy'] >= row['accuracy']
    if not any(cost_condition & accuracy_condition):
        baseline_pareto = pd.concat([baseline_pareto, pd.DataFrame([row.values], columns=subset_all.columns)])
ax.scatter(baseline_pareto['cost'], baseline_pareto['accuracy'], c='purple', marker='x', label='Baseline Optimal', s=100, lw=3)
print_array = baseline_pareto[['k1', 'k2', 'k3', 'k4']].to_numpy()
# Print this array in python format, separated with comma
for row in print_array:
    print(f"    [{row[0]}, {row[1]}, {row[2]}, {row[3]}],")

# Adding annotations to each point in the scatter plot
labels = [f"k1={int(row['k1'])}, k2={int(row['k2'])}, k3={int(row['k3'])}, k4={int(row['k4'])}" for _, row in df.iterrows()]
ax.scatter(selected_df['cost'], selected_df['accuracy'], c='green', label='LLM-Cascading Optimal', s=80)
# Use mplcursors for interactivity
mplcursors.cursor(scatter, hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

if "val" in input_file:
    title = "Codegen HumanEval val set (49 questions) pick@0,1,2,3,4,5,10, averaged at 10 runs, on n x 3090"
else:
    title = "Codegen HumanEval test set (115 questions) pick@0,1,2,3,4,5,10, averaged at 10 runs, on n x 3090"
plt.title(title, fontsize=14)
plt.xlabel("Total cost to run 1000 questions ($)", fontsize=15)
plt.ylabel("Accuracy (%)", fontsize=15)
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()  # Adjusts the layout to fit the title and axis labels
plt.show()
