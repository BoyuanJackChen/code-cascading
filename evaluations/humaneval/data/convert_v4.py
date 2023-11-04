import json

# Read the file and parse the data
data = []
with open('HumanEval_v2.json', 'r') as infile:
    for line in infile:
        data.append(json.loads(line))

# Dump the data to a new JSON file
with open('output_file.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

print("Conversion complete!")