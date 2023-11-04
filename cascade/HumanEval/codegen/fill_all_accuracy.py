import json
import pandas as pd
import multiprocessing
import numpy as np
import os

num_loops = 10
all_pick_at = [0,1,2,3,4,5,10]
test_lines = 1
model_name = "350M"
all_accuracies = np.zeros(num_loops)

for pick_at in all_pick_at:
    for loop in range(num_loops):
        answer_file = f"./{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
        if not os.path.exists(answer_file):
            continue
        # Load the answer file
        with open(answer_file, 'r') as f:
            answer_data = json.load(f)
        output_dict_array = []
        
        # Check if indeed is a key of answer_data[0]
        # if "indeed" in answer_data[0].keys():
        #     continue
        print(f"Working on {model_name}, {pick_at}, {loop}")

        # Create a pandas dataframe with two columns: number and accuracy
        df = pd.DataFrame(columns=["number", "accuracy"])

        multiple_pass = False
        all_keys = answer_data[0].keys()
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
            # if "indeed" in answer_dict.keys():
            #     output_dict_array.append(answer_dict)
            #     continue
            checkpoint = answer_dict["checkpoint"]
            correct = False
            number_str = answer_dict["number"]
            if type(number_str) == str:
                number = int(number_str.split('/')[-1])
            else:
                number = number_str
            if number in df["number"].values:
                continue
            if model_name=="350M" and number == 99 and loop==9 and pick_at==3:
                answer_dict["indeed"] = False
                output_dict_array.append(answer_dict)
                continue
            answer = answer_dict["answer"]
            print(f"On number {number}")
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
            process.join(2)
            if process.is_alive():
                # print("Code took too long to run!")
                process.terminate()
                process.join()  # Ensure termination
                correct = False
            else:
                correct = result_queue.get()
            process.close()
            
            df.loc[len(df)] = [number, int(correct)]
            # print(f"Question {number} is correct: {correct}")
            answer_dict["indeed"] = correct
            if type(answer_dict["number"])==str:
                answer_dict["number"] = int(answer_dict["number"].split('/')[-1])
            output_dict_array.append(answer_dict)

        accuracy = df["accuracy"].mean()
        print(f"Loop {loop} accuracy: {accuracy}")
        all_accuracies[loop] = accuracy
        
        # Write output_dict_array to answer_file
        with open(answer_file, 'w') as f:
            json.dump(output_dict_array, f, indent=4)

mean_accuracy = np.mean(all_accuracies)
mean_accuracy = f"{round(mean_accuracy*100, 1)}%"
print(f"\n\nMean accuracy: {mean_accuracy}")