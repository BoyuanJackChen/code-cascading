import json
import pandas as pd
import multiprocessing
import numpy as np
import os

num_loops = 0
test_lines = 1
model_name = "34B"
pick_at = 0
loop = 0

answer_file = f"./wave_4_greene/{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"

# Load the answer file
with open(answer_file, 'r') as f:
    answer_data = json.load(f)
output_dict_array = []
        
for answer_dict in answer_data:
    answer_dict["answer"] = answer_dict["answer"][1:]
    output_dict_array.append(answer_dict)

# Write output_dict_array to answer_file
with open(answer_file, 'w') as f:
    json.dump(output_dict_array, f, indent=4)