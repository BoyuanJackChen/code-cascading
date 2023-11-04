import numpy as np

# Load in the numpy array from the .npy file
arr = np.load('softmax_0.9_4_4_0.3_transformerscodegen-16B-mono_early_layers.npy')

# Find the average of the array
avg = np.mean(arr)

print(f'The average of the array is {avg}')