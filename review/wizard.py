from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os

stopwords = ["\n\n\n", "\n\n"]
feedback_unspecific = "\n\n# Above is the answer you just generated, but it was actually incorrect. Can you fix it?\n"

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        # Delete "<|endoftext|>" in answer, if it is in it
        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
        answer = answer[:min_i]
        result.append(answer)
    return result

if __name__== "__main__":
    checkpoint = "WizardLM/WizardCoder-Python-34B-V1.0"
    output_file_name = "wizard_output.json"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")
    model.eval()
    
    # Load the questions
    all_questions_dict = json.load(open("../evaluations/humaneval/data/HumanEval_py.json", "r"))

    answer_dict_list = []
    start_time = time.time()
    for question_dict in all_questions_dict:
        number = question_dict["task_id"]
        print(f"\n\n--- On Question {number} ---\n")
        prompt = question_dict["prompt"]
        
        # Get the def line
        clean_promt = ""
        prompt_list = prompt.split("\n")
        for idx, line in enumerate(prompt_list):
            if line.strip().startswith('def'):
                clean_prompt = '\n'.join(prompt_list[:idx+1])
        
        prompt_original = prompt + ""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        start_generating = time.time()
        # generated_ids = model.generate(
        #     input_ids,
        #     use_cache=True,
        #     pad_token_id=tokenizer.eos_token_id,
        #     max_new_tokens=300,
        #     do_sample=True,
        #     top_p=0.95,
        #     temperature=0.2
        # )
        # Greedy search for pass@1 generations
        generated_ids = model.generate(
            input_ids,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            num_beams=1
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        print(f"generated_text is:\n{generated_text[0]}")
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        generated_len = len(decoded_list) - len(input_ids[0])
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt = tokenizer.decode(prompt_ids[0])
        trimmed_text = trim_with_stopwords(generated_text, stopwords, prompt)[0]
        print(f"trimmed_text is:\n{trimmed_text}")
        answer_dict = {'number': number, 
                       'prompt': prompt_original, 
                       'first_answer': trimmed_text}
        
        # New content with feedback
        new_text = prompt_original + trimmed_text
        prompt = new_text + feedback_unspecific + clean_prompt
        print(f"New prompt is:\n{prompt}")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        generated_ids = model.generate(
            input_ids,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            num_beams=1
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        generated_len = len(decoded_list) - len(input_ids[0])
        prompt = tokenizer.decode(input_ids[0])
        print(f"generated new text is:\n{generated_text[0]}")
        trimmed_text = trim_with_stopwords(generated_text, stopwords, prompt)[0]
        print(f"trimmed new text is:\n{generated_text[0]}")
        answer_dict['second_answer'] = trimmed_text
        answer_dict['feedback'] = feedback_unspecific
        answer_dict["test"] = question_dict["test"],
        answer_dict["canonical_solution"] = question_dict["canonical_solution"]
        answer_dict["declaration"] = question_dict["declaration"]
        answer_dict_list.append(answer_dict)
        
        # Update to json file
        if not os.path.exists(output_file_name):
            output_data = [answer_dict]
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
            answer_dict_list = []
        else:
            with open(output_file_name, 'r') as f:
                output_data = json.load(f)
            output_data += answer_dict_list
            with open(output_file_name, 'w') as f:
                json.dump(output_data, f, indent=4)
            answer_dict_list = []
    
    print(f"Time to generate is {time.time() - start_time} seconds")