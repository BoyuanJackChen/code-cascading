import json

# with open('mbpp.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f.readlines()]
#
# for i in range(0, len(data)):
#     index = i
#     one_dict = data[index]
#     print(one_dict.keys())
#     print(f"index is {index}")
#     print(one_dict['text'])
#     print()
#     print(f"{one_dict['code']}")
#     print(one_dict['test_setup_code'])
#     for test in one_dict['test_list']:
#         print(test)
#     input()

with open('sanitized-mbpp.jsonl', 'r') as f:
    data = json.load(f)

# for i in range(0, len(data)):
#     one_dict = data[i]
#     print(one_dict.keys())
#     for k in one_dict.keys():
#         print(f"{k}: {one_dict[k]}")
#     input()

# dump with indent 4
with open('sanitized-mbpp.jsonl', 'w') as f:
    json.dump(data, f, indent=4)