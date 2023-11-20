import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

threshold = 1.0
seed = 7
# Load the datasets
df1 = pd.read_csv(f'./cascade_results/{seed}/{seed}_test_threshold{threshold}.csv')
df2 = pd.read_csv(f'./cascade_results/{seed}/{seed}_pareto_threshold{threshold}.csv')

green_size = 45
purple_size = 50

# Create a scatter plot for the first dataset
plt.figure(figsize=(10, 6))
lighter_purple = (0.7, 0.2, 0.7)
lighter_green = (0.1, 0.6, 0.1)

# Scatter plot objects
scatter_plots = []

# First plot all light blue dots
for _, row in df1.iterrows():
    non_negative_count = sum(n >= 0 for n in [row['k1'], row['k2'], row['k3'], row['k4']])
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

def update_annot(ind, scatter):
    pos = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = f"{df1.iloc[ind['ind'][0]][['k1', 'k2', 'k3', 'k4', 't1', 't2', 't3', 't4']].tolist()}"
    annot.set_text(text)

def hover(event):
    for scatter in scatter_plots:
        vis = annot.get_visible()
        if event.inaxes == scatter.axes:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind, scatter)
                annot.set_visible(True)
                plt.draw()
            else:
                if vis:
                    annot.set_visible(False)
                    plt.draw()

plt.connect("motion_notify_event", hover)

# Overlay the second dataset
for _, row in df2.iterrows():
    color = lighter_green if row['Singular'] == 0 else lighter_purple
    marker = 'o' if row['Singular'] == 0 else 'x'
    size = purple_size if color == lighter_purple else green_size  # Adjust size for both green dots and purple crosses
    # Find the row in df1 with the same k1-k4, t1-t4 values
    row1 = df1.loc[(df1['k1'] == row['k1']) & (df1['k2'] == row['k2']) & (df1['k3'] == row['k3']) & (df1['k4'] == row['k4']) & (df1['t1'] == row['t1']) & (df1['t2'] == row['t2']) & (df1['t3'] == row['t3']) & (df1['t4'] == row['t4'])]
    scatter = plt.scatter(row1['cost'], row1['accuracy'], color=color, marker=marker, s=size, alpha=1.0, linewidths=2)
    scatter_plots.append(scatter)

# Add legends for green dots and purple crosses
plt.scatter([], [], color=lighter_green, label='Cascading Optimal Parameters from Val', s=green_size)
plt.scatter([], [], color=lighter_purple, marker='x', s=purple_size, alpha=0.9, linewidths=2, label='Singular Optimal Parameters from Val')

plt.xlabel('Cost per 1k questions ($)', fontsize=12.5)
plt.ylabel('Accuracy (%)', fontsize=12.5)
plt.title(f'WizardCoder-Python-V1.0 on HumanEval, θ={threshold}', fontsize=14)

# Shift the legend to lower right
plt.legend(loc='lower right')
# Save the plot to pdf
plt.savefig(f'./cascade_results/{seed}_test_threshold{threshold}.pdf', bbox_inches='tight')

plt.show()
