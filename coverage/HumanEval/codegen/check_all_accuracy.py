import json
import pandas as pd
import multiprocessing
import numpy as np
import random

num_loops = 10
pick_at = 0
test_lines = 1
loop = 0
all_accuracies = np.zeros(num_loops)
df = pd.DataFrame(columns=["number", "350M", "2B", "6B", "16B"])
for i in range(1, 165):
    df.loc[len(df)] = [i, 0, 0, 0, 0]

for model_name in ["350M", "2B", "6B", "16B"]:
    answer_file = f"{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
    bad_questions = []
    # Load the answer file
    with open(answer_file, 'r') as f:
        answer_data = json.load(f)

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

    import_lines = "import math\nfrom typing import List\n"
    for i in range(len(answer_data)):
        answer_dict = answer_data[i]
        checkpoint = answer_dict["checkpoint"]
        correct = False
        number_str = answer_dict["number"]
        number = int(number_str.split('/')[-1])
        answer = answer_dict["answer"]
        prompt = answer_dict["prompt"]
        test = answer_dict["test"]
        test = test[test.find("def"):]
        full_code = import_lines + prompt + answer + "\n\n" + test
        
        def code_to_run(result_queue):
            try:
                exec(full_code, globals())
                result_queue.put(True)
            except Exception as e:
                result_queue.put(False)

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
        process.start()
        process.join(3)
        if process.is_alive():
            # print("Code took too long to run!")
            process.terminate()
            process.join()  # Ensure termination
            correct = False
        else:
            correct = result_queue.get()
        process.close()
        
        if "correct" not in answer_dict.keys() and multiple_pass and answer_dict['pass'] < max_pass and not correct:
            continue
        else:
            correct_int = 1 if correct else 0
            df.loc[number, model_name] = correct_int
        print(f"The accuracy of {model_name} is {np.mean(df[model_name])}")

# Write df to local file
df.to_csv("all_greedy_accuracies.csv", index=False)
