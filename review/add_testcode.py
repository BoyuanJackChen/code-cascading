from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os

if __name__== "__main__":
    # Load the questions
    all_questions_dict = json.load(open("./1_0.69h.json", "r"))
    all_data_dict = json.load(open("./16B_no_instruct.json"))
    output_file = "16B_no_instruct_new.json"
    all_output_dict = []

    answer_dict_list = []
    start_time = time.time()
    for i in range(len(all_data_dict)):
        data_dict = all_data_dict[i]
        question_dict = all_questions_dict[i]
        data_dict["test"] = question_dict["test"]
        data_dict["test"] = data_dict["test"][data_dict["test"].find("def"):]
        data_dict["canonical_solution"] = question_dict["canonical_solution"]
        old_second = data_dict["second_answer"]
        if "\"\"\"\n" in old_second:
            new_prompt = old_second.split("\"\"\"\n", 1)[1]
            data_dict['second_answer'] = new_prompt
        all_output_dict.append(data_dict)
    
    # Dump output_dict to json
    with open(output_file, "w") as f:
        json.dump(all_output_dict, f, indent=4)