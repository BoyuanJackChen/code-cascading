import json
import numpy as np
import pandas as pd
import multiprocessing
import os

# Create files in selected
num_loops = 10
pick_at = 10
all_limit_lines = [2,4]
all_actual_pick_at = [0,1,3,5,10]
model_name = "34B"
all_accuracies = np.zeros(num_loops)
import_lines = "import math\nfrom typing import List, Tuple\n"
all_questions_num = list(range(4000,5000))

# Start going through each question
for limit_lines in all_limit_lines:
    for actual_pick_at in all_actual_pick_at:
        for loop in range(num_loops):
            if actual_pick_at == 0:
                answer_file = f"./answer/{model_name}/{model_name}_p0_l0.json"
                testcase_file = f"./testcase/{model_name}/{model_name}_p0_l0.json"
                selected_file = f"./selected/{model_name}/{model_name}_p0_t0_l0.json"
            elif actual_pick_at == 1:
                answer_file = f"./answer/{model_name}/{model_name}_p0_l0.json"
                testcase_file = f"./testcase/{model_name}/{model_name}_p0_l0.json"
                selected_file = f"./selected/{model_name}/{model_name}_p1_t{limit_lines}_l0.json"
            else:
                answer_file = f"./answer/{model_name}/{model_name}_p{pick_at}_l{loop}.json"
                testcase_file = f"./testcase/{model_name}/{model_name}_p{pick_at}_l{loop}.json"
                selected_file = f"./selected/{model_name}/{model_name}_p{actual_pick_at}_t{limit_lines}_l{loop}.json"
            if not os.path.exists(selected_file):
                continue
            print(f"Filling num_ids for {selected_file}...")
            all_selected = []
            
            # Load the answer and testcase files
            with open(answer_file, 'r') as f:
                answer_data = json.load(f)
            with open(testcase_file, 'r') as f:
                testcase_data = json.load(f)
            with open(selected_file, 'r') as f:
                selected_data = json.load(f)

            for number in all_questions_num:
                # Collect the number of ids, so we can calculate time
                num_ids = 0
                selected_dict = selected_data[number-4000]
                
                # Collect all answers for this question
                all_answers = []
                answers_pick_at = max(actual_pick_at+0, 1)   # We still want 1 answer when k=0
                for answer_dict in answer_data:
                    if answer_dict["number"]==number and answers_pick_at>0:
                        num_ids += answer_dict["num_ids"]
                        answers_pick_at -= 1
                
                # Collect all tests for this question
                tests_pick_at = actual_pick_at + 0
                for testcase_dict in testcase_data:
                    if testcase_dict["number"] == number and tests_pick_at>0:
                        num_ids += testcase_dict[f"num_ids_{limit_lines}"]
                        tests_pick_at -= 1
                
                # No need to check if k=0, since there is no testcase
                if actual_pick_at == 0:
                    selected_dict["num_ids"] = int(num_ids)
                    all_selected.append(selected_dict)
                    continue

                selected_dict["num_ids"] = int(num_ids)
                all_selected.append(selected_dict)
                print(f"Question {number}: num_ids: {num_ids})")

            # Write to file
            with open(selected_file, 'w') as f:
                json.dump(all_selected, f, indent=4)