import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import mplcursors

label_size = 18
title_size = 23
legend_size = 15

green_size = 90
purple_size = 95
lightblue_size = 40
darkblue_size = 180
cross_line_width = 3

threshold = 0.9
seed = 1
# Load the datasets
df1 = pd.read_csv(f'./cascade_results/{seed}/{seed}_test_threshold{threshold}.csv')
df2 = pd.read_csv(f'./cascade_results/{seed}/{seed}_pareto_threshold{threshold}.csv')
spec_point = [6.344650799999996*0.969,23.285714285714285]

# Create a scatter plot for the first dataset
plt.figure(figsize=(10, 6))
lighter_purple = (0.7, 0.2, 0.7)
lighter_green = (0.1, 0.6, 0.1)
darker_blue = (0.2, 0.2, 0.5)
light_yellow = (0.9, 0.9, 0.3)

# Create a mapping of points to dataframe rows
point_to_data = {}
for _, row in df1.iterrows():
    point = (row['cost'], row['accuracy'])
    point_to_data[point] = row

# Scatter plot objects
scatter_plots = []

# First plot all light blue dots
for _, row in df1.iterrows():
    non_negative_count = sum(n >= 0 for n in [row['k1'], row['k2'], row['k3']])
    if non_negative_count > 1:
        scatter = plt.scatter(row['cost'], row['accuracy'], color='lightblue', zorder=1)
        scatter_plots.append(scatter)

# Draw grid lines
plt.grid(which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

# Adding hover labels
annot = plt.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="w"),
                     arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind, scatter, df):
    pos = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    data = df.iloc[ind['ind'][0]]
    text = f"k1-k3: {data['k1']}, {data['k2']}, {data['k3']}\n" \
           f"t1-t3: {data['t1']}, {data['t2']}, {data['t3']}\n" \
           f"Cost: {data['cost']}, Accuracy: {data['accuracy']}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

# Instead of the 'hover' function and plt.connect, use mplcursors
cursor = mplcursors.cursor(scatter_plots, hover=True)

@cursor.connect("add")
def on_add(sel):
    # Retrieve the point from the selection
    point = (sel.target[0], sel.target[1])

    # Use the mapping to get the corresponding data
    data = point_to_data.get(point, None)

    # Update the annotation text with additional information
    if data is not None:
        sel.annotation.set_text(
            f"Cost: {round(data['cost'],1)}, Accuracy: {round(data['accuracy'],1)}\n"
            f"k1: {int(data['k1'])}, k2: {int(data['k2'])}, k3: {int(data['k3'])}\n"
            f"t1: {int(data['t1'])}, t2: {int(data['t2'])}, t3: {int(data['t3'])}"
        )

# Draw speculative decoding result in darker blue, using dark blue size
plt.scatter(*spec_point, color=light_yellow, marker='o', s=darkblue_size, alpha=1.0, linewidths=cross_line_width)

# Draw the single-model results and green dots
for _, row in df2.iterrows():
    color = lighter_green if row['Singular'] == 0 else lighter_purple
    marker = 'o' if row['Singular'] == 0 else 'x'
    size = purple_size if color == lighter_purple else green_size
    row1 = df1.loc[(df1['k1'] == row['k1']) & (df1['k2'] == row['k2']) & (df1['k3'] == row['k3']) & (df1['t1'] == row['t1']) & (df1['t2'] == row['t2']) & (df1['t3'] == row['t3'])]
    scatter = plt.scatter(row1['cost'], row1['accuracy'], color=color, marker=marker, s=size, alpha=1.0, linewidths=cross_line_width)
    scatter_plots.append(scatter)

# # Add legends for green dots, purple crosses, and light blue dots
# plt.scatter([], [], color='lightblue', label='All Parameter Combinations', s=lightblue_size)
# plt.scatter([], [], color=lighter_green, label='Cascading Optimal Parameter Combinations from Val', s=green_size)
# plt.scatter([], [], color=lighter_purple, marker='x', s=purple_size, alpha=0.9, linewidths=cross_line_width, label='Singular Optimal Parameter Combinations from Val')
# plt.legend(loc='lower right', fontsize=legend_size)

plt.xlabel('Cost per 1k questions ($)', fontsize=label_size)
plt.ylabel('Accuracy (%)', fontsize=label_size)
plt.title(f'WizardCoder-Python-V1.0 on APPS-Intro, θ={threshold}', fontsize=title_size)

# # Save the plot to pdf
# plt.savefig(f'./cascade_results/{seed}_test_threshold{threshold}.pdf', bbox_inches='tight')
# plt.show()

# Create a figure and a subplot
fig, ax = plt.subplots()
ax.plot([], [])  # Create an empty plot

plt.scatter([], [], color='lightblue', label='All parameter combinations', s=lightblue_size)
plt.scatter([], [], color=lighter_green, label='Cascading optimal parameter combinations from val subset', s=green_size)
plt.scatter([], [], color=lighter_purple, marker='x', s=purple_size, alpha=0.9, linewidths=cross_line_width, label='Singular optimal parameter combinations from val subset')
plt.scatter([], [], color=light_yellow, marker='o', s=darkblue_size, alpha=1.0, label='Speculative decoding')


# Add the legend to the plot
legend = ax.legend(loc='lower right', fontsize=legend_size)

# Hide everything except the legend
ax.axis('off')
legend.set_frame_on(False)

# Show only the legend
plt.savefig(f'./cascade_results/legends.pdf', bbox_inches='tight')
