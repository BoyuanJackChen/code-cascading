import json

# Load sc_apps_test.json
with open('sc_apps_train.json') as json_file:
    data = json.load(json_file)
    
# sort data with key "task_id", in ascending order
data.sort(key=lambda x: x["task_id"])

# dump to sc_apps_train_sorted.json
with open('sc_apps_train_sorted.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)