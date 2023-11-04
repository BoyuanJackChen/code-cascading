import pandas as pd

# Read the CSV file into a Pandas DataFrame
file_path = 'starcoder.csv'
data = pd.read_csv(file_path)

# Count the occurrences of 1's and 0's in the 'correctness' column
correctness_counts = data['correctness'].value_counts()

# Print the counts
print("Count of 1's and 0's in the 'correctness' column:")
print(correctness_counts)
correctness_rate = correctness_counts[1] / (correctness_counts[0] + correctness_counts[1])
print("Correctness rate:", correctness_rate)

unique_value_counts = data['question_number'].value_counts(dropna=False)
print(len(unique_value_counts))

duplicates = data[data['question_number'].duplicated(keep=False)]
print(duplicates)