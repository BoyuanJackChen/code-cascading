import json
import numpy as np
import pandas as pd
import multiprocessing

loop = 0
pick_at = 0
model_name = "34B"

answer_file = f"./answer/{model_name}/{model_name}_p{pick_at}_l{loop}.json"
bad_questions = []
all_correct = []

# Load APPS Dataset
data_file = "../../../evaluations/apps/APPS_zeroshot_for_code_generation.jsonl"

# Load the answer file
with open(answer_file, 'r') as f:
    answer_data = json.load(f)

# Create a pandas dataframe with two columns: number and accuracy
df = pd.DataFrame(columns=["number", "accuracy"])

# [number, prompt, checkpoint, passed, answer, generated_testcode, test]
import_lines = "import random\nimport math\nfrom typing import List, Tuple\n"
for i in range(len(answer_data)):
    answer_dict = answer_data[i]
    correct = False
    number = answer_dict["number"]
    answer = answer_dict["answer"]
    question_dict = {}
    with open(data_file, "r") as f:
        for j, line in enumerate(f):
            if j == number:
                question_dict = json.loads(line)
                break
    # print(question_dict.keys())
    # print(question_dict["task_id"])
    prompt = question_dict["prompt"]
    test = question_dict["test"]
    test += f"\ncheck(solution)"
    # print(test)
    # input()
    
    full_code = import_lines + answer + "\n" + test
    # print(full_code)
    # input()
    
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

# Write to a txt file
output_file = f"accuracy_{model_name}.txt"
with open(output_file, "w") as f:
    f.write(f"Mean accuracy: {mean_accuracy}\n")
