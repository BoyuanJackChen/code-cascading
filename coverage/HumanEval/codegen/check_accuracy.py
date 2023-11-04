import json
import pandas as pd
import multiprocessing

answer_file = "16B_p1_t1_l0.json"
bad_questions = []
# Load the answer file
with open(answer_file, 'r') as f:
    answer_data = json.load(f)

# Create a pandas dataframe with two columns: number and accuracy
df = pd.DataFrame(columns=["number", "accuracy"])
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
import_lines = "import math\nfrom typing import List\n"
for i in range(len(answer_data)):
    answer_dict = answer_data[i]
    checkpoint = answer_dict["checkpoint"]
    correct = False
    if cascade_mode and not answer_dict["passed"] and checkpoint != "16B":
        continue
    number_str = answer_dict["number"]
    number = int(number_str.split('/')[-1])
    if number in df["number"].values:
        continue
    answer = answer_dict["answer"]
    prompt = answer_dict["prompt"]
    test = answer_dict["test"]
    test = test[test.find("def"):]
    full_code = import_lines + prompt + answer + "\n\n" + test

    def code_to_run(result_queue):
        try:
            exec(full_code, globals())
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
        df.loc[len(df)] = [number, int(correct)]

accuracy = df["accuracy"].mean()
print(f"Accuracy: {accuracy}")
print(f"This is the result of {answer_file}")