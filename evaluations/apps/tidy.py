import json
from langdetect import detect

def is_english(text):
    try:
        language = detect(text)
    except:
        return False
    if language == 'en':
        return True
    else:
        return False

# data = []
# with open('test.jsonl', 'r') as f:
#     for line in f:
#         data.append(json.loads(line))
#
# # Dump in indent 4
# with open('test_2.jsonl', 'w') as f:
#     json.dump(data, f, indent=4)

with open('test.jsonl', 'r') as f:
    data = json.load(f)

for i in data:
    if not is_english(i['question']):
        # Remove i from data
        data.remove(i)

# Dump in indent 4
with open('test_eng.jsonl', 'w') as f:
    json.dump(data, f, indent=4)