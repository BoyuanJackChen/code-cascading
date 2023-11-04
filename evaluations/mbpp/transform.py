import json

with open('sanitized-mbpp.json', 'r') as f:
    data = json.load(f)
for i in data:
    print(i)
    input()

# dump with indent 4
with open('sanitized-mbpp.json', 'w') as f:
    json.dump(data, f, indent=4)