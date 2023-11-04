import json
import pandas as pd
import multiprocessing

class TimeoutException(Exception):
    pass

answer_file = "16B_output.json"
bad_questions = []
with open(answer_file, 'r') as f:
    answer_data = json.load(f)

# Create a pandas dataframe with two columns: number and accuracy
df_1 = pd.DataFrame(columns=["number", "accuracy"])
df_2 = pd.DataFrame(columns=["number", "accuracy"])

cascade_mode = False
multiple_pass = False
all_keys = answer_data[0].keys()
if "passed" in all_keys:
    cascade_mode = True
if "pass" in all_keys:
    multiple_pass = True

# Find the biggest pass
if multiple_pass:
    max_pass = 0
    for i in range(len(answer_data)):
        answer_dict = answer_data[i]
        current_pass = answer_dict["pass"]
        if current_pass==1 and max_pass>=1:
            break
        if current_pass > max_pass:
            max_pass = answer_dict["pass"]
    print(f"--- pass@{max_pass} ---")
    

# [number, prompt, checkpoint, passed, answer, generated_testcode, test]
for i in range(len(answer_data)):
    answer_dict = answer_data[i]
    correct = False
    if cascade_mode and not answer_dict["passed"]:
        continue
    number_str = answer_dict["number"]
    number = int(number_str.split('/')[-1])
    if number in df_1["number"].values:
        continue
    answer_1 = answer_dict["first_answer"]
    answer_2 = answer_dict["second_answer"]
    prompt = answer_dict["prompt"]
    test = answer_dict["test"]
    if len(test) == 1:
        test = test[0]
    test = test[test.find("def"):]
    full_code_1 = prompt + answer_1 + "\n\n" + test
    full_code_2 = prompt + answer_2 + "\n\n" + test

    def code_to_run(result_queue):
        try:
            exec(full_code_1, globals())
            print("No error occurred!\n")
            result_queue.put(True)
        except Exception as e:
            print(f"Error occurred: {e}\n")
            result_queue.put(False)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
    process.start()
    process.join(3)
    if process.is_alive():
        print("Code took too long to run!")
        process.terminate()
        correct = False
    else:
        correct = result_queue.get()
    # process.close()
    
    if multiple_pass and answer_dict['pass'] < max_pass and not correct:
        continue
    else:
        print(f"Number {number} is correct: {correct}")
        df_1.loc[len(df_1)] = [number, int(correct)]


for i in range(len(answer_data)):
    answer_dict = answer_data[i]
    correct = False
    if cascade_mode and not answer_dict["passed"]:
        continue
    number_str = answer_dict["number"]
    number = int(number_str.split('/')[-1])
    if number in df_2["number"].values:
        continue
    answer_1 = answer_dict["first_answer"]
    answer_2 = answer_dict["second_answer"]
    prompt = answer_dict["prompt"]
    test = answer_dict["test"]
    if len(test) == 1:
        test = test[0]
    test = test[test.find("def"):]
    full_code_1 = prompt + answer_1 + "\n\n" + test
    full_code_2 = prompt + answer_2 + "\n\n" + test

    def code_to_run(result_queue):
        try:
            exec(full_code_2, globals())
            print("No error occurred!\n")
            result_queue.put(True)
        except Exception as e:
            print(f"Error occurred: {e}\n")
            result_queue.put(False)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
    process.start()
    process.join(5)
    if process.is_alive():
        print("Code took too long to run!")
        process.terminate()
        correct = False
    else:
        correct = result_queue.get()
    process.close()
    
    if multiple_pass and answer_dict['pass'] < max_pass and not correct:
        continue
    else:
        print(f"Number {number} is correct: {correct}")
        df_2.loc[len(df_2)] = [number, int(correct)]
    
accuracy_1 = df_1["accuracy"].mean()
accuracy_2 = df_2["accuracy"].mean()
print(df_1[(df_1['accuracy'] == 0) & (df_2['accuracy'] == 1)].index)
print(f"Accuracy_1: {accuracy_1}; Accuracy_2: {accuracy_2}")
