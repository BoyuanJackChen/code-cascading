import json
import numpy as np
import pandas as pd
import multiprocessing

loop = 0
pick_at = 0
test_lines = 1
model_name = "13B"

answer_file = f"./{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
bad_questions = []
all_correct = []
# Load the answer file
with open(answer_file, 'r') as f:
    answer_data = json.load(f)

# Create a pandas dataframe with two columns: number and accuracy
df = pd.DataFrame(columns=["number", "accuracy"])

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
    # print(f"--- pass@{max_pass} ---")

# [number, prompt, checkpoint, passed, answer, generated_testcode, test]
import_lines = "import math\nfrom typing import List, Tuple\n"
for i in range(len(answer_data)):
    answer_dict = answer_data[i]
    checkpoint = answer_dict["checkpoint"]
    correct = False
    if cascade_mode and not answer_dict["passed"] and checkpoint != "16B":
        continue
    number = answer_dict["number"]
    answer = answer_dict["answer"]
    if answer.startswith("   ") and not answer.startswith("    "):
        answer = " " + answer
    if answer.startswith("     ") and not answer.startswith("      "):
        answer = answer[1:]
    prompt = answer_dict["prompt"]
    test = answer_dict["test"]
    
    # Find the last line that starts with "def "
    def_line = ""
    lines = answer.split("\n")
    for line in reversed(lines):
        if line.startswith("def "):
            def_line = line
            break
    def_name = def_line.split(" ")[1].split("(")[0]
    test = test[test.find("def "):]
    test = test + f"\ncheck({def_name})"
    
    # full_code = import_lines + prompt + answer + "\n" + test
    full_code = import_lines + answer + "\n" + test
    
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
        process.join()  # Ensure termination
        correct = False
    else:
        correct = result_queue.get()
    process.close()
    
    if correct:
        all_correct.append(number)
    df.loc[len(df)] = [number, int(correct)]
    print(f"Question {number} is correct: {correct}")

accuracy = df["accuracy"].mean()
mean_accuracy = accuracy
mean_accuracy = f"{round(mean_accuracy*100, 1)}%"
print(f"\n\nMean accuracy: {mean_accuracy}")