import json
import pandas as pd
import multiprocessing
import numpy as np
import os
import re

num_loops = 1
all_pick_at = [0,1,2,3,4,5,10]
all_pick_at = [10]
test_lines = 1
model_name = "34B"
all_accuracies = np.zeros(num_loops)

def trim_outside_triple_quotes(text):
    # Split the text by triple quotes
    parts = text.split('"""')
    # If the length is odd, there is a mismatch in triple quotes
    for i in range(len(parts)):
        # For parts not inside triple quotes, apply the trimming logic
        if i % 2 == 0:
            for newline_pattern in ["\n\n", "\n \n", "\n  \n", "\n#", "\ndef"]:
                idx = parts[i].find(newline_pattern)
                if idx >= 0:
                    parts[i] = parts[i][:idx]
            # Check for a newline followed by a character that is not a space or tab
            match = re.search(r'\n[^\s\t]', parts[i])
            if match:
                parts[i] = parts[i][:match.start() + 1]
    # Join the parts back together
    return '"""'.join(parts)

for pick_at in all_pick_at:
    for loop in range(num_loops):
        answer_file = f"./{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
        actual_answer_file = f"./{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}_actual.json"
        # actual_answer_file = answer_file
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
            checkpoint = answer_dict["checkpoint"]
            correct = False
            number = answer_dict["number"]
            if number in df["number"].values or number < 200:
                continue
            answer = answer_dict["answer"]
            # print(number)
            # print(answer)
            # print()
            if answer.startswith("   ") and not answer.startswith("    "):
                answer = " " + answer
            if answer.startswith("     "):
                answer = answer[1:]
                
            # Trim
            answer = trim_outside_triple_quotes(answer)
            # def_index = answer.rfind("\ndef ")
            # if def_index >= 0 or def_index < -1:
            #     answer = answer[:def_index]
            
            prompt = answer_dict["prompt"]
            test = answer_dict["test"]
            def_line = prompt[prompt.find("def "):]
            # Remove all empty spaces at the end of def_line
            while def_line.endswith(" "):
                def_line = def_line[:-1]
            full_code = import_lines + def_line + answer + "\n" + test
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
            # print(f"Question {number} is correct: {correct}")
            answer_dict["indeed"] = correct
            output_dict_array.append(answer_dict)
            # print(full_code)
            # print(number, correct)
            # input()

        accuracy = df["accuracy"].mean()
        print(f"Loop {loop} accuracy: {accuracy}")
        all_accuracies[loop] = accuracy
        
        # Write output_dict_array to answer_file
        with open(actual_answer_file, 'w') as f:
            json.dump(output_dict_array, f, indent=4)

mean_accuracy = np.mean(all_accuracies)
mean_accuracy = f"{round(mean_accuracy*100, 1)}%"
print(f"\n\n{model_name} pick@{pick_at} mean accuracy: {mean_accuracy}")