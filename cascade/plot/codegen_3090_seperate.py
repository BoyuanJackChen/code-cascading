import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# Data
model_16B= [[0.32, 32.3],
            [0.59, 29.1],
            [0.71, 32.1],
            [0.80, 35.2],
            [0.89, 34.5],
            [1.29, 39.4]]
model_6B = [[0.20, 28.0],
            [0.33, 25.4],
            [0.39, 28.0],
            [0.43, 28.9],
            [0.49, 30.9],
            [0.70, 34.0]]
model_2B = [[0.18, 25.6],
            [0.28, 20.4],
            [0.33, 22.7],
            [0.36, 22.4],
            [0.39, 26.0],
            [0.52, 27.3]]
model_350M=[[0.16, 14.0],
            [0.29, 11.0],
            [0.24, 12.0],
            [0.27, 13.6],
            [0.30, 13.5],
            [0.38, 15.8]]

models = {
    '16B: 67G, 5': model_16B,
    '6B: 30G, 3': model_6B,
    '2B: 12G, 2': model_2B,
    '350M: 2G, 1': model_350M
}

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

plt.title("Codegen HumanEval pick@ 1,2,3,4,5,10, averaged at 10 runs, on n x 3090, method 2", fontsize=16)
plt.xlabel("Time taken for answer and test generation on full dataset (h)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
