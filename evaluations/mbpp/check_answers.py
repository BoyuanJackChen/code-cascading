import json
import multiprocessing
import concurrent
import pytest
import os
import csv

model_size = "6B"
with open('sanitized-mbpp.jsonl', 'r') as f:
    prompt_data = json.load(f)

with open(f'answers/starcoder_k1.json', 'r') as f:
    answer_data = json.load(f)

output_file_name = f'./correctness/starcoder.csv'
output_file_exists = os.path.exists(output_file_name)
if output_file_exists:
    os.remove(output_file_name)
with open(output_file_name, 'a+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['question_number', 'correctness'])

def run_code(code):
    global_namespace = globals().copy()
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError:
        return False
    try:
        exec(code, global_namespace)
    except (AssertionError, AttributeError, TypeError, RecursionError, ZeroDivisionError, NameError, IndexError,
            ValueError, TimeoutError):
        return False
    return True


def test_code_exec(code, timeout=5):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_code, code)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            return False


def change_function_name(answer, test):
    answer_fun_name = answer.split("def ")[1].split("(")[0]
    # replace the string between 'set(' and ')' with the test function name
    if "set(" in test:
        test = test.replace(test.split("set(")[1].split("(")[0], answer_fun_name)
    elif "math.isclose(" in test:
        test = test.replace(test.split("math.isclose(")[1].split("(")[0], answer_fun_name)
    else:
        test = test.replace(test.split("assert ")[1].split("(")[0], answer_fun_name)
    if "math." in test:
        test = "import math; " + test
    return test


for i in range(len(answer_data)):
    result = None
    answer_dict = answer_data[i]
    prompt_dict = prompt_data[i]
    assert answer_dict["task_id"] == prompt_dict["task_id"]
    number = answer_dict["task_id"]
    if (model_size=="350M" and number in [123]) or \
            (model_size=="2B" and number in [246]) or \
            (model_size=="6B" and number in [739]):
        with open(output_file_name, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([number, 0])
        continue
    answer = answer_dict["answer"]
    test_list = prompt_dict["test_list"]
    test_code = ""
    for test in test_list:
        test_code += "\n" + test
    full_ori = answer + test_code
    test_code = ""
    for test in test_list:
        test = change_function_name(answer, test)
        test_code += "\n" + test
    full_testing_code = answer + test_code

    try:
        result = test_code_exec(full_testing_code)
    except:
        print(f"full ori:")
        print(full_ori)
        print()
        print(full_testing_code)
        print()
    print(number, result)
    result_idx = 1 if result else 0
    with open(output_file_name, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([number, result_idx])