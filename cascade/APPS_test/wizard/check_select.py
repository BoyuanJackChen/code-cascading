import json
import numpy as np
import pandas as pd
import multiprocessing
import os
from datasets import load_dataset

num_loops = 10
pick_at = 10
all_limit_lines = [2,4]
all_actual_pick_at = [1,3,5,10]
model_name = "7B"
all_accuracies = np.zeros(num_loops)
import_lines = "import math\nfrom typing import List, Tuple\n"
all_questions_num = list(range(4000,5000))

# Mkdir
if not os.path.exists(f"./selected"):
    os.mkdir(f"./selected")
if not os.path.exists(f"./selected/{model_name}"):
    os.mkdir(f"./selected/{model_name}")

def find_max_product(matrix):
    max_product = -1
    max_indices = (-1, -1)
    max_answer_num = 0
    max_test_num = 0
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    for a in range(matrix.shape[0]):
        for t in range(matrix.shape[1]):
            if matrix[a][t] == 1:
                product = row_sums[a] * col_sums[t]
                if product > max_product:
                    max_product = product
                    max_indices = (a, t)
                    max_answer_num = row_sums[a]
                    max_test_num = col_sums[t]
    return max_product, max_indices, max_answer_num, max_test_num

# Start going through each question
for limit_lines in all_limit_lines:
    for actual_pick_at in all_actual_pick_at:
        for loop in range(num_loops):
            if actual_pick_at == 1:
                answer_file = f"./answer/{model_name}/{model_name}_p0_l0.json"
                testcase_file = f"./testcase/{model_name}/{model_name}_p0_l0.json"
                selected_file = f"./selected/{model_name}/{model_name}_p1_t{limit_lines}_l0.json"
            else:
                answer_file = f"./answer/{model_name}/{model_name}_p{pick_at}_l{loop}.json"
                testcase_file = f"./testcase/{model_name}/{model_name}_p{pick_at}_l{loop}.json"
                selected_file = f"./selected/{model_name}/{model_name}_p{actual_pick_at}_t{limit_lines}_l{loop}.json"
            if os.path.exists(selected_file):
                continue
            all_selected = []
            all_correct = []
            
            # Load the answer and testcase files
            with open(answer_file, 'r') as f:
                answer_data = json.load(f)
            with open(testcase_file, 'r') as f:
                testcase_data = json.load(f)

            for number in all_questions_num:
                # Collect all answers for this question
                all_answers = []
                answers_pick_at = actual_pick_at + 0
                for answer_dict in answer_data:
                    if answer_dict["number"]==number and answers_pick_at>0:
                        answer = answer_dict["answer"]
                        all_answers.append(answer)
                        answers_pick_at -= 1
                
                # Collect all tests for this question
                all_generated_tests = []
                tests_pick_at = actual_pick_at + 0
                for testcase_dict in testcase_data:
                    if testcase_dict["number"] == number and tests_pick_at>0:
                        this_test = testcase_dict["answer"]
                        testlines = this_test.split("\n")
                        for j in range(min(limit_lines, len(testlines))):
                            if testlines[j].startswith("assert"):
                                all_generated_tests.append(testlines[j])
                        tests_pick_at -= 1
                
                # Initiate the correctness stats
                correct_stats = np.zeros([len(all_answers),len(all_generated_tests)], np.int32)
                
                # Check the correctness for each combination
                def code_to_run(a, t, import_lines, answer, generated_test, result_queue):
                    full_code = import_lines + answer + "\n" + generated_test
                    try:
                        exec(full_code, globals())
                        result_queue.put((a, t, True))
                    except Exception as e:
                        result_queue.put((a, t, False))
                processes = []
                result_queue = multiprocessing.Queue()
                
                # Start all processes without waiting for them to complete
                for a in range(len(all_answers)):
                    for t in range(len(all_generated_tests)):
                        answer = all_answers[a]
                        generated_test = all_generated_tests[t]
                        process = multiprocessing.Process(target=code_to_run, args=(a, t, import_lines, answer, generated_test, result_queue))
                        process.start()
                        processes.append(process)

                # Impose a 1-second time limit on each process
                for process in processes:
                    process.join(1)  # Kill infinite loops in 1 second
                    if process.is_alive():
                        process.terminate()
                        process.join()

                # After all processes are done or terminated, retrieve results from the queue
                while not result_queue.empty():
                    a, t, correct = result_queue.get()
                    if correct:
                        correct_stats[a][t] += 1

                # Close all processes
                for process in processes:
                    process.close()
                
                max_product, indices, max_answer_num, max_test_num = find_max_product(correct_stats)
                selected_answer = all_answers[indices[0]]
                selected_test = all_generated_tests[indices[1]]
                selected_dict = {
                    "number": number,
                    "max_answer_num": int(max_answer_num),
                    "max_test_num": int(max_test_num),
                    "total_product": int(len(all_answers)*len(all_generated_tests)),
                    "answer": selected_answer,
                    "test": selected_test
                }
                all_selected.append(selected_dict)
                print(f"Question {number}: Max product: {max_product}; indices: {indices}, ({max_answer_num}, {max_test_num})")
                # print(correct_stats)

            # Write to file
            with open(selected_file, 'w') as f:
                json.dump(all_selected, f, indent=4)