import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

threshold = 1.0
# Load the datasets
df1 = pd.read_csv(f'./cascade_results/3_test_threshold{threshold}.csv')
df2 = pd.read_csv(f'./cascade_results/3_pareto_threshold{threshold}.csv')

green_size = 40
purple_size = 50

# Create a scatter plot for the first dataset
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df1['cost'], df1['accuracy'], color='lightblue', label='All Combinations')

# Draw grid lines
plt.grid(which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

# Adding hover labels
annot = plt.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="w"),
                     arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    pos = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = f"k1-k3, t1-t3 values: {df1.iloc[ind['ind'][0]][['k1', 'k2', 'k3', 't1', 't2', 't3']].tolist()}"
    annot.set_text(text)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == scatter.axes:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            plt.draw()
        else:
            if vis:
                annot.set_visible(False)
                plt.draw()

plt.connect("motion_notify_event", hover)

# Overlay the second dataset
for _, row in df2.iterrows():
    color = 'green' if row['Singular'] == 0 else 'purple'
    marker = 'o' if row['Singular'] == 0 else 'x'
    size = purple_size if color == 'purple' else green_size  # Adjust size for both green dots and purple crosses
    # Find the row in df1 with the same k1-k3, t1-t3 values
    row1 = df1.loc[(df1['k1'] == row['k1']) & (df1['k2'] == row['k2']) & (df1['k3'] == row['k3']) & (df1['t1'] == row['t1']) & (df1['t2'] == row['t2']) & (df1['t3'] == row['t3'])]
    plt.scatter(row1['cost'], row1['accuracy'], color=color, marker=marker, s=size)

# Add legends for green dots and purple crosses
plt.scatter([], [], color='green', label='Combinations Optimal Settings', s=green_size)
plt.scatter([], [], color='purple', marker='x', s=purple_size, label='Singular Model Optimal Settings')

plt.xlabel('Cost ($)', fontsize=13)
plt.ylabel('Accuracy (%)', fontsize=13)
plt.title(f'WizardCoder-Python-V1.0 on HumanEval, pick@0,1,3,5,10, testlines=2,4, threshold={threshold}, test set', fontsize=14)

# Shift the legend to lower right
plt.legend(loc='lower right')

plt.show()
