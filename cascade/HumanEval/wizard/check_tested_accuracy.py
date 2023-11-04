import json
import pandas as pd
import multiprocessing
import numpy as np

num_loops = 10
pick_at = 10
test_lines = 1
model_name = "13B"
all_accuracies = np.zeros(num_loops)
# Create a pandas dataframe with two columns: number and accuracy
df = pd.DataFrame(columns=["number", "accuracy", "True positive", "True negative", "False positive", "False negative"])

for loop in range(num_loops):
    answer_file = f"./wave_1_jubail_sep/{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
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

    # [number, prompt, checkpoint, passed, answer, generated_testcode, test]
    import_lines = "import math\nfrom typing import List\n"
    for i in range(len(answer_data)):
        answer_dict = answer_data[i]
        checkpoint = answer_dict["checkpoint"]
        correct = False
        if cascade_mode and not answer_dict["passed"] and checkpoint != "16B":
            continue
        number_str = answer_dict["number"]
        number = int(number_str.split('/')[-1])
        # if (loop==2 and number==78):
        #     continue
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
            # print("Code took too long to run!")
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
        df.loc[len(df)] = [number, int(correct), true_positive, true_negative, false_positive, false_negative]
        print(f"Question {number} is correct: {correct}; tested result: {tested_result}")

    accuracy = df["accuracy"].mean()
    true_positive = df["True positive"].mean()
    true_negative = df["True negative"].mean()
    false_positive = df["False positive"].mean()
    false_negative = df["False negative"].mean()
    print(f"Loop {loop} accuracy: {accuracy}")

accuracy = round(df["accuracy"].mean()*100, 1)
true_positive = round(df["True positive"].mean()*100, 1)
true_negative = round(df["True negative"].mean()*100, 1)
false_positive = round(df["False positive"].mean()*100, 1)
false_negative = round(df["False negative"].mean()*100, 1)
print(f"\n\nMean accuracy: {accuracy}; {true_positive} {true_negative} {false_positive} {false_negative}%")