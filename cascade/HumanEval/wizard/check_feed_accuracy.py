import json
import pandas as pd
import multiprocessing
import numpy as np

num_loops = 1
pick_at = 1
test_lines = 1
model_name = "13B"
all_accuracies = np.zeros(num_loops)
all_accuracies_feed = np.zeros(num_loops)
for loop in range(num_loops):
    answer_file = f"./wave_1_jubail_sep/{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
    feed_answer_file = f"./old_feed/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
    bad_questions = []
    
    # Load the answer file
    with open(answer_file, 'r') as f:
        answer_data = json.load(f)
    with open(feed_answer_file, 'r') as f:
        feed_answer_data = json.load(f)

    # Create a pandas dataframe with two columns: number and accuracy
    df = pd.DataFrame(columns=["number", "accuracy"])
    df_feed = pd.DataFrame(columns=["number", "accuracy"])

    cascade_mode = False
    multiple_pass = False
    all_keys = answer_data[0].keys()
    if "passed" in all_keys:
        cascade_mode = True
    if "pass" in all_keys:
        multiple_pass = True

    # [number, prompt, checkpoint, passed, answer, generated_testcode, test]
    import_lines = "import math\nfrom typing import List\n"
    true_positive = np.zeros(len(feed_answer_data))
    true_positive_feed = np.zeros(len(feed_answer_data))
    for i in range(len(feed_answer_data)):
        # Feed
        answer_dict = feed_answer_data[i]
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
        df_feed.loc[len(df_feed)] = [number, int(correct)]
        print(f"Question {number} is correct: {correct}")
        if answer_dict["correct"] and correct:
            true_positive_feed[i] += 1
        
        # Normal
        answer_dict = answer_data[number]
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
        df.loc[len(df)] = [number, int(correct)]
        print(f"Question {number} is correct: {correct}")
        if answer_dict["correct"] and correct:
            true_positive[i] += 1
        #     if true_positive_feed[i] == 0:
        #         print(prompt)
        #         print(answer)
        #         print()
        #         print(feed_answer_data[i]["answer"])
        #         print(answer_data[number]["generated_testcode"])
        #         print(feed_answer_data[i]["generated_testcode"])
        #         input()

    accuracy = df["accuracy"].mean()
    all_accuracies[loop] = accuracy
    accuracy_feed = df_feed["accuracy"].mean()
    all_accuracies_feed[loop] = accuracy_feed
    print(f"Loop {loop} normal accuracy: {accuracy}, feed accuracy: {accuracy_feed}")
    
mean_accuracy = np.mean(all_accuracies)
mean_accuracy = f"{round(mean_accuracy*100, 1)}%"
feed_accuracy = np.mean(all_accuracies_feed)
feed_accuracy = f"{round(feed_accuracy*100, 1)}%"
mean_true_positive = np.mean(true_positive)
mean_true_positive = f"{round(mean_true_positive*100, 1)}%"
mean_true_positive_feed = np.mean(true_positive_feed)
mean_true_positive_feed = f"{round(mean_true_positive_feed*100, 1)}%"
print(f"\n\nMean accuracy: {mean_accuracy}, Feed accuracy: {feed_accuracy}")
print(f"Mean true positive: {mean_true_positive}, Feed true positive: {mean_true_positive_feed}")