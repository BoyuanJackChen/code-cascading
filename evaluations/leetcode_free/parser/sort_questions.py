import json

# Read in the original JSON data from file
with open('data.json', 'r') as f:
    data = json.load(f)

# Sort the list of dictionaries based on the "number" key
sorted_data = sorted(data, key=lambda x: x['number'])

# Write the sorted data to a new JSON file
with open('sorted_data.json', 'w') as f:
    json.dump(sorted_data, f, indent=4)