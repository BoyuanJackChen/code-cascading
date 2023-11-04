import json
import pandas as pd
import multiprocessing
import numpy as np

k1 = 2
k2 = -1
k3 = 2
num_loops_1 = 10
num_loops_2 = 1
num_loops_3 = 10
test_lines = 1
all_final_accuracy = []
question_num_1 = []
question_num_2 = []
question_num_3 = []
model_runs = pd.DataFrame(columns=["7B", "13B", "34B"]) 

for loop_1 in range(num_loops_1):
    for loop_2 in range(num_loops_2):
        for loop_3 in range(num_loops_3):
            # Create a pandas dataframe with two columns: number and accuracy
            df_1 = pd.DataFrame(columns=["number", "accuracy", "True positive", "True negative", "False positive", "False negative"])
            df_2 = pd.DataFrame(columns=["number", "accuracy", "True positive", "True negative", "False positive", "False negative"])
            df_3 = pd.DataFrame(columns=["number", "accuracy", "True positive", "True negative", "False positive", "False negative"])
            df = pd.DataFrame(columns=["number", "accuracy", "model"])

            answer_file_1 = f"./wave_1_jubail_sep/7B/7B_p{k1}_t{test_lines}_l{loop_1}.json"
            answer_file_2 = f"./wave_1_jubail_sep/13B/13B_p{k2}_t{test_lines}_l{loop_2}.json"
            answer_file_3 = f"./wave_1_jubail_sep/34B/34B_p{k3}_t{test_lines}_l{loop_3}.json"
            bad_questions = []
            # Load the answer file

            with open(answer_file_3, 'r') as f:
                answer_data_3 = json.load(f)
            import_lines = "import math\nfrom typing import List\n"
            k2_questions = []
            k3_questions = []

            # Process k1 model
            # [number, prompt, checkpoint, passed, answer, generated_testcode, test]
            if k1 < 0:
                k2_questions = list(range(164))
            else:
                with open(answer_file_1, 'r') as f:
                    answer_data_1 = json.load(f)
                question_num_1.append(len(answer_data_1))
                for i in range(len(answer_data_1)):
                    answer_dict = answer_data_1[i]
                    checkpoint = answer_dict["checkpoint"]
                    correct = False
                    number_str = answer_dict["number"]
                    number = int(number_str.split('/')[-1])
                    answer = answer_dict["answer"]
                    if answer.startswith("   ") and not answer.startswith("    "):
                        answer = " " + answer
                    prompt = answer_dict["prompt"]
                    test = answer_dict["test"]
                    test = test[test.find("def"):]
                    tested_result = answer_dict["correct"]
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
                    process.join(2)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                        correct = False
                    else:
                        correct = result_queue.get()
                    process.close()

                    true_positive = 1 if tested_result and correct else 0
                    true_negative = 1 if not tested_result and not correct else 0
                    false_positive = 1 if tested_result and not correct else 0
                    false_negative = 1 if not tested_result and correct else 0
                    df_1.loc[len(df_1)] = [number, int(correct), true_positive, true_negative, false_positive, false_negative]
                    # print(f"Question {number} is correct: {correct}; tested result: {tested_result}")
                    if not tested_result:
                        k2_questions.append(number)
                    else:
                        df.loc[len(df)] = [number, int(correct), 1]
                
            # Process k2 model
            if k2 < 0:
                k3_questions = k2_questions if len(k2_questions) > 0 else list(range(164))
            else:
                with open(answer_file_2, 'r') as f:
                    answer_data_2 = json.load(f)
                question_num_2.append(len(k2_questions))
                for i in k2_questions:
                    answer_dict = answer_data_2[i]
                    checkpoint = answer_dict["checkpoint"]
                    correct = False
                    number_str = answer_dict["number"]
                    number = int(number_str.split('/')[-1])
                    answer = answer_dict["answer"]
                    if answer.startswith("   ") and not answer.startswith("    "):
                        answer = " " + answer
                    prompt = answer_dict["prompt"]
                    test = answer_dict["test"]
                    test = test[test.find("def"):]
                    tested_result = answer_dict["correct"]
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
                    process.join(2)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                        correct = False
                    else:
                        correct = result_queue.get()
                    process.close()
                    
                    true_positive = 1 if tested_result and correct else 0
                    true_negative = 1 if not tested_result and not correct else 0
                    false_positive = 1 if tested_result and not correct else 0
                    false_negative = 1 if not tested_result and correct else 0
                    df_2.loc[len(df_2)] = [number, int(correct), true_positive, true_negative, false_positive, false_negative]
                    # print(f"Question {number} is correct: {correct}; tested result: {tested_result}")
                    if not tested_result:
                        k3_questions.append(number)
                    else:
                        df.loc[len(df)] = [number, int(correct), 2]

            # Process k3 model
            if k3 < 0:
                pass
            else:
                question_num_3.append(len(k3_questions))
                for i in k3_questions:
                    answer_dict = answer_data_3[i]
                    checkpoint = answer_dict["checkpoint"]
                    correct = False
                    number_str = answer_dict["number"]
                    number = int(number_str.split('/')[-1])
                    answer = answer_dict["answer"]
                    if answer.startswith("   ") and not answer.startswith("    "):
                        answer = " " + answer
                    prompt = answer_dict["prompt"]
                    test = answer_dict["test"]
                    test = test[test.find("def"):]
                    tested_result = answer_dict["correct"]
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
                    process.join(2)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                        correct = False
                    else:
                        correct = result_queue.get()
                    process.close()
                    
                    true_positive = 1 if tested_result and correct else 0
                    true_negative = 1 if not tested_result and not correct else 0
                    false_positive = 1 if tested_result and not correct else 0
                    false_negative = 1 if not tested_result and correct else 0
                    df_3.loc[len(df_3)] = [number, int(correct), true_positive, true_negative, false_positive, false_negative]
                    df.loc[len(df)] = [number, int(correct), 3]
            

            # Sort df on number and print fully
            df = df.sort_values(by=["number"])
            # print(df.to_string(index=False))

            # accuracy = df["accuracy"].mean()
            # true_positive = df["True positive"].mean()
            # true_negative = df["True negative"].mean()
            # false_positive = df["False positive"].mean()
            # false_negative = df["False negative"].mean()
            # print(f"Loop {loop} accuracy: {accuracy}")

            accuracy = round(df["accuracy"].mean()*100, 1)
            print(f"\n\nMean accuracy: {accuracy}; ")
            all_final_accuracy.append(accuracy)
            
            model_runs.loc[len(model_runs)] = [len(df_1), len(df_2), len(df_3)]

print(all_final_accuracy)
print(f"Average accuracy: {np.mean(np.array(all_final_accuracy))}")
print(f"Model 1 question num: {np.mean(np.array(question_num_1))}")
print(f"Model 2 question num: {np.mean(np.array(question_num_2))}")
print(f"Model 3 question num: {np.mean(np.array(question_num_3))}")

# Process cost data
column_averages = model_runs.mean()
print(column_averages)
