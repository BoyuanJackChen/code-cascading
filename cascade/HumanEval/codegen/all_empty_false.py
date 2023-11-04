import json
import pandas as pd
import multiprocessing
import numpy as np
import os

num_loops = 10
test_lines = 1
model_name = "2B"
all_accuracies = np.zeros(num_loops)
for pick_at in [1,2,3,4,5,10]:
    for loop in range(num_loops):
        answer_file = f"{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
        if not os.path.exists(answer_file):
            continue
        # Load the answer file
        with open(answer_file, 'r') as f:
            answer_data = json.load(f)

        for i in range(len(answer_data)):
            answer_dict = answer_data[i]
            if answer_dict["generated_testcode"] != "":
                continue
            number = answer_dict["number"]
            if type(number) == str:
                number = int(number.split('/')[-1])
            answer_data[i]["correct"] = False

        with open(answer_file, 'w') as f:
            json.dump(answer_data, f, indent=4)
