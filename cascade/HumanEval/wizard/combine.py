import json

file_7B = "./wave_1_jubail_sep/7B/7B_p5_t1_l0.json"
file_13B = "./13B_feed/13B_p1_t1_l0.json"
file_34B = "./34B_feed/34B_p0_t1_l0.json"

data_7B = json.load(open(file_7B, 'r'))
data_13B = json.load(open(file_13B, 'r'))
data_34B = json.load(open(file_34B, 'r'))
seen_questions = []

final_data = []
for question_dict in data_34B:
    final_data.append(question_dict)
    number = question_dict["number"]
    seen_questions.append(number)

for question_dict in data_13B:
    number = question_dict["number"]
    if number in seen_questions:
        continue
    final_data.append(question_dict)
    seen_questions.append(number)

for question_dict in data_7B:
    number = question_dict["number"]
    if number in seen_questions:
        continue
    final_data.append(question_dict)
    seen_questions.append(number)
    
# sort final_data based on "number"
final_data = sorted(final_data, key=lambda x: x["number"].split("/")[1])

with open("./combined_feed.json", 'w') as f:
    json.dump(final_data, f, indent=4)


    