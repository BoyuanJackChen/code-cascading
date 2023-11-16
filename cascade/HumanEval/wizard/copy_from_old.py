import json
import pandas as pd
import multiprocessing
import numpy as np
import os
import re

all_num_loops = 10
all_pick_at = [0,1,3,5,10]
all_testlines = [0,2,4]
model_name = "13B"

# Load HumanEval Dataset
all_questions_dict = json.load(open("../../../evaluations/humaneval/data/HumanEval_py.json", "r"))

for pick_at in all_pick_at:
    for testlines in all_testlines:
        num_loops = all_num_loops if pick_at>1 else 1
        all_accuracies = np.zeros(num_loops)
        for loop in range(num_loops):
            answer_file = f"./selected/{model_name}/{model_name}_p{pick_at}_t{testlines}_l{loop}.json"
            old_answer_file = f"./selected_old/{model_name}/{model_name}_p{pick_at}_t{testlines}_l{loop}.json"
            actual_answer_file = f"./selected/{model_name}/{model_name}_p{pick_at}_t{testlines}_l{loop}_actual.json"
            actual_answer_file = answer_file
            if not os.path.exists(answer_file):
                continue
            if not os.path.exists(old_answer_file):
                old_answer_file = f"./selected_old/{model_name}/{model_name}_p1_t2_l{loop}.json"
            # Load the answer file
            with open(answer_file, 'r') as f:
                answer_data = json.load(f)
            with open(old_answer_file, 'r') as f:
                old_answer_data = json.load(f)
            output_dict_array = []
            
            print(f"Working on {model_name}, {pick_at}, {loop}")

            for i in range(len(answer_data)):
                answer_dict = answer_data[i]
                old_answer_dict = old_answer_data[i]
                question_dict = all_questions_dict[i]
                old_question_dict = all_questions_dict[i]
                answer_dict["indeed"] = old_answer_dict["indeed"]
                output_dict_array.append(answer_dict)
            
            # Write output_dict_array to answer_file
            with open(actual_answer_file, 'w') as f:
                json.dump(output_dict_array, f, indent=4)
            