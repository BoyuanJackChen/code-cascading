import json
import pandas as pd
import multiprocessing
import numpy as np
import os
from datasets import load_dataset

num_loops = 1
all_pick_at = [0]
test_lines = 1
model_name = "34B"
all_accuracies = np.zeros(num_loops)

# Load APPS-test dataset
all_questions_dict = load_dataset("codeparrot/apps", split="test")
number_key = "problem_id"
prompt_key = "question"
difficulty_key = "difficulty"
solutions_key = "solutions"
test_key = "input_output"
startercode_key = "starter_code"

def contains_input_assignment(canonical_solution):
    canonical_solution = canonical_solution.replace("\\n", "\n")
    canonical_solution_lines = canonical_solution.split('\n')
    for line in canonical_solution_lines:
        if "=" in line and "input()" in line:
            return True
    return False

for pick_at in all_pick_at:
    for loop in range(num_loops):
        answer_file = f"./{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
        output_file = f"./{model_name}/{model_name}_p{pick_at}_t{test_lines}_l{loop}_indeed.json"
        if not os.path.exists(answer_file):
            continue
        # Load the answer file
        with open(answer_file, 'r') as f:
            answer_data = json.load(f)
        output_dict_array = []
        
        # Check if indeed is a key of answer_data[0]
        if "indeed" in answer_data[0].keys():
            continue
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
        for i in range(0, len(answer_data)):
            answer_dict = answer_data[i]
            checkpoint = answer_dict["checkpoint"]
            correct = False
            number = answer_dict["number"]
            if number in df["number"].values:
                continue
            answer = answer_dict["answer"]
            answer = answer.replace("input()", "input_string")
            if answer.startswith("   ") and not answer.startswith("    "):
                answer = " " + answer
            
            prompt_dict = all_questions_dict[number]
            prompt = prompt_dict[prompt_key]
            input_output = prompt_dict[test_key]
            
            parsed_data = json.loads(input_output)
            inputs = parsed_data["inputs"]
            outputs = parsed_data["outputs"]
            canonical_collage = prompt_dict[solutions_key][2:-2]
            canonical_solution_list = canonical_collage.split('", "')
            shortest_len = 100000
            if len(canonical_solution_list) > 0:
                canonical_idx = 0
                for idx in range(len(canonical_solution_list)):
                    if len(canonical_solution_list[idx]) < shortest_len and contains_input_assignment(canonical_solution_list[idx]):
                        shortest_len = len(canonical_solution_list[idx])
                        canonical_idx = idx
                canonical_solution = canonical_solution_list[canonical_idx]
                canonical_solution = canonical_solution.replace("\\n", "\n")
            else:
                canonical_solution = ""
            
            is_wrong = False
            test = ""
            # Loop through both arrays
            for (input_str, output_str) in zip(inputs, outputs):
                if type(input_str) is not str:
                    test += f"assert(solution({input_str}) == {output_str})\n"
                    continue
                # Remove all Replace '\n' with '\\n' in input and output
                if input_str.endswith("\n"):
                    input_str = input_str[:-1]
                if output_str.endswith("\n"):
                    output_str = output_str[:-1]
                # Check if output_str can be converted to integer
                try:
                    output_int = int(output_str)
                except:
                    output_int = None
                input_str = input_str.replace("\n", "\\n")
                output_str = f"'{output_str}'" if output_int is None else str(output_int)
                test += f"assert(solution('{input_str}') == {output_str})\n"
            full_code = import_lines + answer + "\n\n" + test
            # print(f"Question {number}\n prompt:\n{prompt}\nfull code:\n{full_code}\ncanonical solution:\n{canonical_solution}")
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
            print(f"Question {number} is correct: {correct}")
            answer_dict["indeed"] = correct
            if type(answer_dict["number"])==str:
                answer_dict["number"] = int(answer_dict["number"].split('/')[-1])
            output_dict_array.append(answer_dict)
            
        accuracy = df["accuracy"].mean()
        # Get the mean accuracy of number < 4000
        df_intro = df[df["number"] > 4000]
        intro_accuracy = df_intro["accuracy"].mean()
        df_interview = df[df["number"] < 3000]
        interview_accuracy = df_interview["accuracy"].mean()
        df_competition = df[(df["number"] > 3000) & (df["number"] < 4000)]
        competition_accuracy = df_competition["accuracy"].mean()
        print(f"Loop {loop} accuracy: {accuracy}; intro: {intro_accuracy}; interview: {interview_accuracy}; competition: {competition_accuracy}")
        all_accuracies[loop] = accuracy
        
        # Write output_dict_array to answer_file
        with open(output_file, 'w') as f:
            json.dump(output_dict_array, f, indent=4)

mean_accuracy = np.mean(all_accuracies)
mean_accuracy = f"{round(mean_accuracy*100, 1)}%"
print(f"\n\nMean accuracy: {mean_accuracy}")