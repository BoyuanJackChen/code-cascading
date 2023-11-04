import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# Data
model_34B= [[0.31, 58.5],
            [0.61, 56.2],
            [1.15, 59.7],
            [0.83, 61.3],
            [1.05, 62.2],
            [1.33, 64.6]]
model_13B =[[0.15, 47.6],
            [0.28, 51.2],
            [0.53, 53.2],
            [0.39, 54.2],
            [0.52, 57.0],
            [0.75, 57.8]]
model_7B = [[0.12, 44.5],
            [0.21, 48.8],
            [0.32, 50.1],
            [0.25, 52.6],
            [0.34, 54.1],
            [0.40, 55.3]]

models = {
    '34B: 126G, 2': model_34B,
    '13B: 55G, 1': model_13B,
    '7B: 30G, 1': model_7B,
}

# Annotations
annotations = ["p2", "p3", "p4", "p5"]

# Plot
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red', 'purple']  # Define a color palette

for color, (name, data) in zip(colors, models.items()):
    data = np.array(data)
    x, y = data[:, 0], data[:, 1]

    # Create linear interpolation
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")

    # Create x_new data
    x_new = np.linspace(min(x), max(x), 300)
    y_new = f(x_new)

    # Plot curves with increased linewidth and data points with matching color
    plt.plot(x_new, y_new, label=name, lw=2.5, color=color)
    plt.scatter(x, y, color=color, marker='o', s=50)  # s is the size of the markers
    
    # Annotate only the specified data points
    for xi, yi, anno in zip(x[1:5], y[1:5], annotations):  # Indexing 1:5 to select 2nd, 3rd, 4th, and 5th points
        offset = (-20 if color == 'green' else 10)  # Change offset for the green line (except for p5)
        plt.annotate(anno, (xi, yi), textcoords="offset points", xytext=(0, offset), ha='center')

plt.title("WizardCoder HumanEval pick@ 1,2,3,4,5,10, averaged at 10 runs, on n x a100, method 2", fontsize=16)
plt.xlabel("Time taken for answer and test generation on full dataset (h)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()