import json

# Load sc_apps_test.json
with open('sc_apps_test.json') as json_file:
    data = json.load(json_file)

for answer_dict in data:
    if answer_dict["task_id"] == 20:
        print(answer_dict["answer"])