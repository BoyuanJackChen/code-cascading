import json
from collections import Counter

# read in the json file
with open('350M_k10_leetcode_answers.json', 'r') as f:
    data = json.load(f)

# sort the list of dictionaries by "number" and "pass"
sorted_data = sorted(data, key=lambda x: (x['number'], x['pass']))

# remove duplicate entries based on "number" and "pass"
unique_data = []
deleted_entries = []
for item in sorted_data:
    key = (item['number'], item['pass'])
    if key not in deleted_entries:
        unique_data.append(item)
        deleted_entries.append(key)
    else:
        deleted_entries.append(key)

# count the number of appearances of each number
number_counts = Counter(item['number'] for item in unique_data)

# print the numbers with more than 10 appearances
print('Numbers with more than 10 appearances:', [number for number, count in number_counts.items() if count > 10])

# save the sorted list to a new json file
with open('350M_k10.json.json', 'w') as f:
    json.dump(unique_data, f, indent=4)
