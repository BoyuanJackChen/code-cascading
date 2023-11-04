import json

# load a json file
with open('train.json', 'r') as f:
    train_data = json.load(f)
with open('test.json', 'r') as f:
    test_data = json.load(f)
with open('dev.json', 'r') as f:
    dev_data = json.load(f)
print(f"train_data length is {len(train_data)}")
print(f"test_data length is {len(test_data)}")
print(f"dev_data length is {len(dev_data)}")

with open('test_indent4.json', 'w') as f:
    json.dump(test_data, f, indent=4)

with open('train_indent4.json', 'w') as f:
    json.dump(train_data, f, indent=4)

# for i in range(0, len(test_data)):
#     index = i
#     one_dict = train_data[index]
#     print(one_dict.keys())
#     for k in one_dict.keys():
#         print(f"{k}: {one_dict[k]}")
#     # if 'student' in one_dict['text'] and 'adult' in one_dict['text'] and '120' in one_dict['text']:
#     #     for k in one_dict.keys():
#     #         print(one_dict[k])
#     #         print()
#     input()

# Dump
# # all task_id in test data += 19209
# for i in range(0, len(test_data)):
#     test_data[i]['task_id'] += 19209
# # all task_id in dev data += 19209 + 2822
# for i in range(0, len(dev_data)):
#     dev_data[i]['task_id'] += 19209 + 1883
#
# full_data = train_data + test_data + dev_data
# # dump full data
# with open('mathqa-python.json', 'w') as f:
#     json.dump(full_data, f, indent=4)
