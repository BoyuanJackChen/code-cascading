import pandas as pd

target_file = "16B_k1.csv"
paradigm_file = "350M_k1.csv"

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(target_file)
paradigm_data = pd.read_csv(paradigm_file)

# Remove repetitive entries in data with pandas
data = data.drop_duplicates(subset=['question_number', 'correctness'], keep='first')
paradigm_data = paradigm_data.drop_duplicates(subset=['question_number', 'correctness'], keep='first')

# Sort data and paradigm_data by question_number
data = data.sort_values(by=['question_number'])
paradigm_data = paradigm_data.sort_values(by=['question_number'])

# find missing numbers in data
missing_numbers = []
paradigm_numbers = list(paradigm_data['question_number'])
target_numbers = list(data['question_number'])
for number in paradigm_numbers:
    if number not in target_numbers:
        missing_numbers.append(number)
print(missing_numbers)

# find missing numbers in paradigm_data
missing_numbers = []
for number in target_numbers:
    if number not in paradigm_numbers:
        missing_numbers.append(number)
print(missing_numbers)

# Write the sorted data to a new CSV file
# paradigm_data.to_csv(paradigm_file, index=False)
print(len(data), len(paradigm_data))

# find duplicate question_number in data
duplicate_numbers = []
for number in target_numbers:
    if target_numbers.count(number) > 1:
        duplicate_numbers.append(number)
print(duplicate_numbers)

# Remove rows in data where correctness is '2'
data = data[data['correctness'] != 2]
data = data[data['correctness'] != '2']
data.to_csv(target_file, index=False)