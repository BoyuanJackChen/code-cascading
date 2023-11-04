import json

# load a json file
with open('train.json', 'r') as f:
    train_data = json.load(f)

for i in range(0, len(train_data)):
    index = i
    one_dict = train_data[index]
    # print(one_dict.keys())
    for k in one_dict.keys():
        print()
        print(f"{k}: {one_dict[k]}")
    input()