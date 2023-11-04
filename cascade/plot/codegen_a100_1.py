import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# Data
model_16B= [[0.24, 32.3],
            [0.61, 28.7],
            [0.50, 31.9],
            [0.69, 34.3],
            [0.77, 35.2],
            [1.00, 37.9]]
model_6B = [[0.18, 28.0],
            [0.37, 25.8],
            [0.34, 28.0],
            [0.43, 30.2],
            [0.45, 30.1],
            [0.60, 32.8]]
model_2B = [[0.18, 25.6],
            [0.28, 20.4],
            [0.32, 22.4],
            [0.36, 23.7],
            [0.39, 24.5],
            [0.51, 26.5]]
model_350M=[[0.17, 14.0],
            [0.21, 10.7],
            [0.24, 12.4],
            [0.28, 12.7],
            [0.30, 13.3],
            [0.38, 13.7]]

models = {
    '16B: 67G, 1': model_16B,
    '6B: 30G, 1': model_6B,
    '2B: 12G, 1': model_2B,
    '350M: 2G, 1': model_350M
}

# Annotations
annotations = ["p1", "p2", "p3", "p4", "p5", "p10"]
annotate_models = ["16B: 67G, 1", "6B: 30G, 1"]

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

    # Plot curves with increased linewidth and data points with smaller size and matching color
    plt.plot(x_new, y_new, label=name, lw=2.5, color=color)
    plt.scatter(x, y, color=color, marker='o', s=50)  # s is the size of the markers
    
    # Annotating each data point for specified models
    if name in annotate_models:
        for xi, yi, anno in zip(x[1:4], y[1:4], annotations[1:4]):
            plt.annotate(anno, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')

plt.title("Codegen HumanEval pick@ 1,2,3,4,5,10, averaged at 10 runs, on 1 x a100, method 1", fontsize=16)
plt.xlabel("Time taken for answer and test generation on full dataset (h)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
