import json
import pandas as pd
import multiprocessing
import numpy as np

pick_at = 0
test_lines = 1
loop = 0
df = pd.DataFrame(columns=["number", "7B", "13B", "34B"])
for i in range(1, 165):
    df.loc[len(df)] = [i, 0, 0, 0]

for model_name in ["7B", "13B", "34B"]:
    answer_file = f"{model_name}_p{pick_at}_t{test_lines}_l{loop}.json"
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

    import_lines = "import math\nfrom typing import List, Tuple\n"
    for i in range(len(answer_data)):
        answer_dict = answer_data[i]
        checkpoint = answer_dict["checkpoint"]
        correct = False
        if cascade_mode and not answer_dict["passed"] and checkpoint != "16B":
            continue
        number = answer_dict["number"]
        answer = answer_dict["answer"]
        if answer.startswith("   ") and not answer.startswith("    "):
            answer = " " + answer
        if answer.startswith("     ") and not answer.startswith("      "):
            answer = answer[1:]
        prompt = answer_dict["prompt"]
        test = answer_dict["test"]
        
        # Find the last line that starts with "def "
        def_line = ""
        lines = answer.split("\n")
        for line in reversed(lines):
            if line.startswith("def "):
                def_line = line
                break
        def_name = def_line.split(" ")[1].split("(")[0]
        test = test[test.find("def "):]
        test = test + f"\ncheck({def_name})"
        
        # full_code = import_lines + prompt + answer + "\n" + test
        full_code = import_lines + answer + "\n" + test
        
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
        
        correct_int = 1 if correct else 0
        df.loc[number, model_name] = correct_int
    print(f"The accuracy of {model_name} is {np.mean(df[model_name])}")

# Write df to local file
df.to_csv("all_greedy_accuracies.csv", index=False)
