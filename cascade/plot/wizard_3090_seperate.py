import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# Data
model_34B= [[0.51, 58.5],
            [1.00, 56.2],
            [1.18, 59.7],
            [1.36, 61.3],
            [1.55, 62.2],
            [1.92, 64.6]]
model_13B = [[0.21, 47.6],
            [0.42, 51.2],
            [0.45, 53.2],
            [0.50, 54.2],
            [0.54, 57.0],
            [0.75, 57.8]]
model_7B = [[0.13, 44.5],
            [0.25, 48.8],
            [0.27, 50.1],
            [0.30, 52.6],
            [0.35, 54.1],
            [0.47, 55.3]]

models = {
    '34B: 145G, 8': model_34B,
    '13B: 56G, 4': model_13B,
    '7B: 30G, 2': model_7B,
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

plt.title("WizardCoder HumanEval pick@ 1,2,3,4,5,10, averaged at 10 runs, on n x 3090, method 2", fontsize=16)
plt.xlabel("Time taken for answer and test generation on full dataset (h)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
