import json

answer_data = json.load(open('2B_k1.json', 'r'))
paradigm_data = json.load(open('../answers/2B_k1.json', 'r'))
print(f"Original length: {len(answer_data)}")
all_answer_questions = [item['number'] for item in answer_data]
all_paradigm_questions = [item['number'] for item in paradigm_data]

# Find repetitions
repetitions = []
for i in range(len(all_answer_questions)):
    if all_answer_questions[i] in all_answer_questions[i+1:]:
        repetitions.append(all_answer_questions[i])
print(len(repetitions), repetitions)

# Remove repetitions
non_rep_results = []
for i in range(len(answer_data)):
    if answer_data[i]['number'] in repetitions:
        repetitions.remove(answer_data[i]['number'])
        continue
    else:
        non_rep_results.append(answer_data[i])
print(len(non_rep_results))

# Find missing questions
missing_questions = []
for i in range(len(all_paradigm_questions)):
    if all_paradigm_questions[i] not in all_answer_questions:
        missing_questions.append(all_paradigm_questions[i])
print(len(missing_questions), missing_questions)

# Sort non_rep_results by question number
non_rep_results.sort(key=lambda x: x['number'])

# Write
with open('2B_k1.json', 'w') as f:
    json.dump(non_rep_results, f, indent=4)