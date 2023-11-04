import json
import pandas as pd
import multiprocessing
import numpy as np
import random
from itertools import combinations, combinations_with_replacement, product, permutations

data_folder = "./"
model_1 = "1B"
model_2 = "3B"
model_3 = "15B"
N = 1000
set_size = 164
df_throughput = pd.read_csv("./starcoder_he_throughput.csv").fillna(method='ffill')

for seed in [3,7,9,13,15]:
    random.seed(seed)
    all_questions = list(range(0,set_size))
    selected_numbers = random.sample(range(1, set_size+1), 49)
    val_numbers = [num-1 for num in selected_numbers]
    test_numbers = [num for num in all_questions if num not in selected_numbers]
    # selected_numbers = unselected_numbers
    output_file_name_val = f"./{seed}_val.csv"
    output_file_name_test = f"./{seed}_test.csv"
    num_loops = 10
    test_lines = 1

    import_lines = "import math\nfrom typing import List\n"
    c = 0.02
    c_col = 'Cost on full run ($)'
    t_col = 'time per batch (s)'
    nm_col = 'Num GPUs Occupied'
    b_col = 'max batch'

    df_result = pd.DataFrame(columns=["k1", "k2", "k3", "cost", "accuracy"])

    # Get all k1-kn combos
    all_k_values = [-1] + list(df_throughput['k'])
    all_k_values = [int(value) for value in all_k_values]
    all_k_values = [-1,0,1,2,3,4,5,10]
    combs = list(product(all_k_values, repeat=3))

    def is_valid_combination(comb):
        # Exclude combinations where all entries are -1
        if set(comb) == {-1}:
            return False
        
        # Exclude combinations where an early element is 0, and a later element has a non-negative value.
        after_zero = False
        for val in comb:
            if after_zero and val >= 0:
                return False
            if val == 0:
                after_zero = True
        return True

    # Generate all permutations for each combination and filter them based on the validation function
    all_k_combos = [perm for comb in combs for perm in set(permutations(comb)) if is_valid_combination(perm)]

    # Remove duplicates by converting the list to a set and back to a list
    all_k_combos = list(set(all_k_combos))
    all_k_combos = sorted(all_k_combos, key=lambda x: (x[0], x[1], x[2]))

    for selected_numbers, output_file_name in zip([val_numbers, test_numbers], [output_file_name_val, output_file_name_test]):
        for (k1, k2, k3) in all_k_combos:
            only_one = k1<=1 and k2<=1 and k3<=1
            for loop in range(num_loops):
                # Load initial data and cost values
                loop_1 = loop if k1>1 else 0
                loop_2 = loop if k2>1 else 0
                loop_3 = loop if k3>1 else 0
                answer_file_1 = f"./{data_folder}/{model_1}/{model_1}_p{k1}_t{test_lines}_l{loop_1}.json"
                answer_file_2 = f"./{data_folder}/{model_2}/{model_2}_p{k2}_t{test_lines}_l{loop_2}.json"
                answer_file_3 = f"./{data_folder}/{model_3}/{model_3}_p{k3}_t{test_lines}_l{loop_3}.json"
                t_1 = df_throughput.loc[(df_throughput['k'] == k1) & (df_throughput['Model'] == model_1)][t_col].values[0] if k1>=0 else 0
                t_2 = df_throughput.loc[(df_throughput['k'] == k2) & (df_throughput['Model'] == model_2)][t_col].values[0] if k2>=0 else 0
                t_3 = df_throughput.loc[(df_throughput['k'] == k3) & (df_throughput['Model'] == model_3)][t_col].values[0] if k3>=0 else 0
                nm_1 = df_throughput.loc[(df_throughput['k'] == k1) & (df_throughput['Model'] == model_1)][nm_col].values[0] if k1>=0 else 0
                nm_2 = df_throughput.loc[(df_throughput['k'] == k2) & (df_throughput['Model'] == model_2)][nm_col].values[0] if k2>=0 else 0
                nm_3 = df_throughput.loc[(df_throughput['k'] == k3) & (df_throughput['Model'] == model_3)][nm_col].values[0] if k3>=0 else 0
                b_1 = df_throughput.loc[(df_throughput['k'] == k1) & (df_throughput['Model'] == model_1)][b_col].values[0] if k1>=0 else 1
                b_2 = df_throughput.loc[(df_throughput['k'] == k2) & (df_throughput['Model'] == model_2)][b_col].values[0] if k2>=0 else 1
                b_3 = df_throughput.loc[(df_throughput['k'] == k3) & (df_throughput['Model'] == model_3)][b_col].values[0] if k3>=0 else 1
                num_questions_1 = 0
                num_questions_2 = 0
                num_questions_3 = 0
                final_combined_answer = []
                all_questions_left = selected_numbers + []
                df_accuracy = pd.DataFrame(columns=["number", "correct"])
                
                # Create actual answer file
                if k1>=0:
                    with open(answer_file_1, 'r') as f:
                        answer_1 = json.load(f)
                    for answer_dict in answer_1:
                        number = int(answer_dict["number"].split("/")[-1])
                        if number in all_questions_left:
                            num_questions_1 += 1
                            if (k2<0 and k3<0) or answer_dict["correct"]:
                                final_combined_answer.append(answer_dict)
                                all_questions_left.remove(number)
                if k2>=0:
                    with open(answer_file_2, 'r') as f:
                        answer_2 = json.load(f)
                    for answer_dict in answer_2:
                        number = int(answer_dict["number"].split("/")[-1])
                        if number in all_questions_left:
                            num_questions_2 += 1
                            if k3<0 or answer_dict["correct"]:
                                final_combined_answer.append(answer_dict)
                                all_questions_left.remove(number)
                if k3>=0:
                    with open(answer_file_3, 'r') as f:
                        answer_3 = json.load(f)
                    for answer_dict in answer_3:
                        number = int(answer_dict["number"].split("/")[-1])
                        if number in all_questions_left:
                            num_questions_3 += 1
                            final_combined_answer.append(answer_dict)
                            all_questions_left.remove(number)
                
                # Calculate combination cost
                total_cost = N*c*(num_questions_1*t_1*nm_1/b_1 + num_questions_2*t_2*nm_2/b_2 + num_questions_3*t_3*nm_3/b_3) / (3600*len(selected_numbers))
                total_cost = round(total_cost, 7)
                    
                # Calculate combination accuracy
                for i in range(len(final_combined_answer)):
                    indeed = final_combined_answer[i]["indeed"]
                    correct = 1 if indeed else 0
                    df_accuracy.loc[len(df_accuracy)] = [number, correct]
                total_accuracy = round(df_accuracy["correct"].mean()*100, 1)
                
                # Add to df_result
                df_result.loc[len(df_result)] = [k1, k2, k3, total_cost, total_accuracy]
                print(f"k1: {k1}, k2: {k2}, k3: {k3}, cost: {total_cost}, accuracy: {total_accuracy}")
                if only_one:
                    break

        # Write df_result
        avg_df = df_result.groupby(['k1', 'k2', 'k3']).agg({'cost':'mean', 'accuracy':'mean'}).reset_index()
        # Convert k1, k2, and k3 to integers
        avg_df['k1'] = avg_df['k1'].astype(int)
        avg_df['k2'] = avg_df['k2'].astype(int)
        avg_df['k3'] = avg_df['k3'].astype(int)
        avg_df.to_csv(output_file_name, index=False)