from datasets import load_dataset
import json

all_questions_dict = load_dataset("codeparrot/apps", split="test")
print(all_questions_dict)
starcoder = load_dataset('json', data_files=r'C:\Users\syf_y\OneDrive\Desktop\transformers-v4.29\evaluations\apps\answers\starcoder_k1.json', split='train')
print(starcoder)

def execute_code(code_string, func_name, params):
    # Define the execution context
    exec_context = {}

    # Convert params to a string
    params_str = ', '.join(map(str, params))

    # Append a call to the function at the end of the function definition
    call = f"\nresult = {func_name}({params_str})"
    code_string += call

    # Execute the function definition and call
    try:
        exec(code_string, exec_context)
    except SyntaxError as e:
        return "syntax error"
    
    except Exception as e:
        #print(f"Unexpected error: {e}")
        # Handle other types of errors
        return "unexpected error"
    
    # If the key 'result' is not found in the dictionary, return a specific error message
    return exec_context.get('result', "Function did not return a result.")




def compare_result(result,output):
    if result == "syntax error" or result == "unexpected error" or result != output:
        return 0
    else:
        return 1

def get_function_name(answer):
    firstline = answer.split("\n")[0]
    func = firstline.split("def ")[1]
    func_name = func.split("(")[0]
    return func_name

# Assuming all_questions_dict['test'] and starcoder['train'] are instances of 'Dataset' class
check_list = [("task_id","correctness")]
t=0
result_list = [("task_id","result")]

for ans in starcoder:
    if ans['task_id'] == 14 or ans['task_id'] == 333 or ans['task_id'] == 367 or ans['task_id'] == 2346 or ans['task_id'] == 2403 or ans['task_id'] == 2605 or ans['task_id'] ==2703 or ans['task_id'] == 515 or ans['task_id'] == 591 or ans['task_id'] == 789 or ans['task_id'] == 942 or ans['task_id'] ==1325 or ans['task_id']==1533 or ans['task_id'] == 1632 or ans['task_id'] == 1841 or ans['task_id'] == 3029 or ans['task_id'] == 4296 or ans['task_id'] == 4553 or ans['task_id'] == 4871:
            cur_tup = (ans['task_id'],0)
            check_list.append(cur_tup)
            continue
    for q in all_questions_dict:
        if ans['task_id'] == q['problem_id']:
            print(ans['task_id'])
            input_output = q['input_output']
            input_output = json.loads(input_output)
            input = input_output['inputs']
            output = input_output['outputs']
            answer = ans['answer']
         
            func_name = get_function_name(answer)
            result = execute_code(answer,func_name,input)
            result_list.append((ans['task_id'], result))
            if t==3:
                '''
                print("input is "+ str(input))
                print("------------------------")
                print("output is "+str(output))
                print("------------------------")
                print("result is "+str(result))
                print("------------------------")
                '''
                print("prompt is:")
                print(ans['prompt'])
                print("---------------------------------------------------------------------------------------------------------------------------------")
                print("question is:")
                print(q['question'])
                print(result)
                print("input is "+ str(input))
                print("------------------------")
                print("output is "+str(output))
                print("------------------------")
                print("result is "+str(result))
                print("------------------------")
                break
            
            compare = compare_result(result,output)
            check_list.append((ans['task_id'], compare))
            break
    if t==100:
        break
    t+=1

print("------------------------")
print(check_list)
# Open the file in write mode ('w')
with open(r"C:\Users\syf_y\OneDrive\Desktop\transformers-v4.29\evaluations\apps\check_list.txt", "w") as f:
    for tup in check_list:
        # Convert each tuple to a string and then write it to the file
        f.write(str(tup) + "\n")

with open(r"C:\Users\syf_y\OneDrive\Desktop\transformers-v4.29\evaluations\apps\result_list.txt", "w") as f:
    for tup in result_list:
        # Convert each tuple to a string and then write it to the file
        f.write(str(tup) + "\n")
