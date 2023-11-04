import json
import jsonlines
import re
import sys
import gc
sys.set_int_max_str_digits(500000000)  # increase the limit to 5000 digits

  
# Opening JSON file with 'utf-8' encoding
with open('HumanEvalPlus.json', 'r', encoding='utf-8') as f:
    # returns JSON object as a dictionary
    tasks = json.load(f)
    t=0
    for task in tasks:
        '''
        if t != 160:
            t+=1
            continue
        '''
        print("case"+str(t))
        function_name = task['entry_point']
        code_body = task['canonical_solution']
        prompt = task['prompt']    
        function_signature = re.search(r'\((.*?)\)', prompt).group(1)
        parameter_names = re.findall(r'\b(\w+)\b', function_signature)     
        data_type_strings = {'int', 'float', 'str', 'list', 'dict', 'tuple', 'set','List','Tuple'}
        filtered_names = [name for name in parameter_names if name not in data_type_strings]
        
        parameters = ','.join(filtered_names)

        function_string = f"def {function_name}({parameters}):\n    {code_body}"
        prompt_string =""
        if t==32:
            filtered_names = ['xs']
            parameters = ','.join(filtered_names)
            function_string = f"def {function_name}({parameters}):\n    {code_body}"
            prompt_string = "import math\n\n\ndef poly(xs,x):\n return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\n"
            function_string = prompt_string + function_string 

        if t==115 or t==32:
            function_string ="import math\n" + function_string
        exec(function_string)
        # Call the function with inputs from "base_input" and store the base_outputs

        base_outputs = []
    
        
        e=0
        for input_case in task['base_input']:
            input_str = ""
            pars = []
            # add quatation marks to the string inputs
            for par in input_case:
                if isinstance(par, (str)):
                    par = repr(par)
                pars.append(par)
            # concatenate all the inputs
            input_str = ', '.join(map(str, pars))  # Join list elements with commas

            
            cur_base_output = eval(f"{function_name}({input_str})")
            base_outputs.append(cur_base_output)
            e+=1
        task['base_output'] = base_outputs
        

    #------------------------------------------for plus input-------------------------------------------------
        plus_outputs = []
        task160_plus_input = []
        e=0
        for input_case in task['plus_input']:
            if t==160:
                if e != 303 and e!= 439 and e!= 451:
                    task160_plus_input.append(input_case)
                else:
                    e+=1
                    continue
            input_str = ""
            pars = []
            # add quatation marks to the string inputs
            for par in input_case:
                if isinstance(par, (str)):
                    par = repr(par)
                pars.append(par)
            # concatenate all the inputs
            input_str = ', '.join(map(str, pars))  # Join list elements with commas
            cur_plus_output = eval(f"{function_name}({input_str})")
            plus_outputs.append(cur_plus_output)
            
            e+=1
        if t == 160:
            task['plus_input'] = task160_plus_input
        task['plus_output'] = plus_outputs
        special = []
        if t==160:
           print("160! 303")
           print(type(plus_outputs[303]))
           plus_outputs[303] = str(plus_outputs[303])
           task['plus_output'] = special
        
        
        with jsonlines.open('HumanEvalPlusWithOutputs.jsonl', 'a') as writer:
            try:
                writer.write(task)
                print("completed task" + str(t))
            except:
                print(f"Error writing task {t}: {e}")
        
        del function_string
        del prompt_string
        del parameters
        del base_outputs
        del plus_outputs
        t+=1
    # Call the garbage collector to free up memory
        gc.collect()
