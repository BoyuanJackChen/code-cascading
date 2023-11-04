import re
import numpy as np
import os

# Initialize an empty list to store the times
times = []
pick_at = 10
test_lines = 1
model_name = "34B"


def extract_data(second_last_line):
    pattern = r"This is the result of (\w+[BM]), pick@(\d+), testlines (\d+)"
    matches = re.match(pattern, second_last_line)
    if matches:
        return [matches.group(1), int(matches.group(2)), int(matches.group(3))]
    else:
        return []


files = os.listdir()
# Loop through each file
for file in files:
    if file.endswith('.out'):
        with open(file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1].strip()
            if last_line.startswith("This is the result of "):
                extracted = extract_data(last_line)
                if len(extracted)==3 and extracted[0] == model_name and extracted[1] == pick_at and extracted[2] == test_lines:
                    for line in lines:
                        match = re.search(r'Loop \d+, total time taken: (\d+\.\d+) seconds', line)
                        if match:
                            times.append(float(match.group(1)))
            
# Calculate mean and standard deviation using numpy
mean_time = np.mean(times)
std_dev_time = np.std(times)

print(f"Mean Time: {round(mean_time, 0)} seconds")
print(f"Standard Deviation: {std_dev_time} seconds")
print(f"{model_name}, pick@{pick_at}, testlines {test_lines}")
