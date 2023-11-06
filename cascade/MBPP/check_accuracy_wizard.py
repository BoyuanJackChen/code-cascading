import json
import pandas as pd
import multiprocessing

answer_file = "7B_mbpp.txt"
bad_questions = []
# import_lines = "import math\nfrom typing import List\n"

# Load the answer file into string
with open(answer_file, 'r') as file:
    answers_str = file.read()
answer_data = answers_str.split("\"], [\"")
answer_data[0] = answer_data[0][3:]
answer_data[-1] = answer_data[-1][:-3]

# Load MBPP Dataset
all_questions_dict = []
with open("../../evaluations/mbpp/mbpp_test_wizard.jsonl", 'r') as file:
    for line in file:
        json_line = json.loads(line)
        all_questions_dict.append(json_line)

# print(f"{answer_data}")
# print(all_questions_dict)
# input()

# Create a pandas dataframe with two columns: number and accuracy
df = pd.DataFrame(columns=["number", "accuracy"])

cascade_mode = False
multiple_pass = False

for question_dict, answer_ori in zip(all_questions_dict, answer_data):
    correct = False
    number = question_dict["task_id"]
    if number in df["number"].values:
        continue
    answer = str(answer_ori)
    answer = answer.replace("\\r\\n", "\n")
    answer = answer.replace("\\r", "")
    # answer = answer[answer.find("\n#"):]
    test_list = question_dict["test_list"]
    test = "\n".join(test_list)
    full_code = answer + "\n" + test
    def code_to_run(result_queue):
        try:
            exec(full_code, globals())
            result_queue.put(True)
        except Exception as e:
            result_queue.put(False)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
    process.start()
    process.join(1)
    if process.is_alive():
        # print("Code took too long to run!")
        process.terminate()
        process.join()
        correct = False
    else:
        correct = result_queue.get()
    process.close()
    
    # print(full_code)
    # print(f"Number {number} is correct: {correct}")
    # input()
    df.loc[len(df)] = [number, int(correct)]
    

accuracy = round(df["accuracy"].mean()*100, 2)
print(len(df))
print(f"Accuracy: {accuracy}%")
print(f"This is the result of {answer_file}")