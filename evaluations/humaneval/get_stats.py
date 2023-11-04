from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CodeGenModel
import torch
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--loaded_model', type=str, default='Salesforce/codegen-16B-mono')
parser.add_argument('--num_layers', type=int, default=34)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_samples_per_task', type=int, default=1)
parser.add_argument('--beam_width', type=int, default=4)
parser.add_argument('--num_beam_groups', type=int, default=4)
parser.add_argument('--beam_diversity_rate', type=float, default=0.3)
FLAGS = parser.parse_args()

eos_token = 50256
stop_words = ["\n\n"]
problems = read_problems()


def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i >= 5:
                    min_i = i
        answer = answer[:min_i]
        result.append(answer)
    return result[0]


def main(args):
    loaded = args.loaded_model
    model_name = loaded.split('/')[-1]
    num_layers = args.num_layers
    num_samples_per_task = args.num_samples_per_task
    tokenizer = AutoTokenizer.from_pretrained(loaded, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(loaded, device_map="auto")
    all_appearance = np.zeros((53000,1200), dtype=np.int8)

    beam_width = args.beam_width
    num_beam_groups = args.num_beam_groups
    beam_diversity_rate = args.beam_diversity_rate

    def generate_one_completion(this_prompt):
        this_input_ids = tokenizer(this_prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(
            this_input_ids.to('cuda'),
            max_new_tokens=200,
            eos_token_id=eos_token,
            pad_token_id=eos_token,
            use_cache=False,
            num_beams=beam_width,
            num_beam_groups=num_beam_groups,
            diversity_penalty=beam_diversity_rate,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)

        # Get the start and end index in decoded list
        start_index = 0
        end_index = len(decoded_list)
        for i in range(len(decoded_list)):
            if decoded_list[i] == "\n" and decoded_list[i + 1] == "    " and decoded_list[i + 2] == '"""' \
                    and decoded_list[i + 3] == "\n":
                start_index = i + 4
                break
        for i in range(start_index, len(decoded_list)):
            if decoded_list[i] == "\n\n":
                end_index = i
                break
            elif decoded_list[i] == "\n" and decoded_list[i + 1] == "\n":
                end_index = i + 1
                break
        trimmed_text = trim_with_stopwords(generated_text, stop_words, this_prompt)
        decoded_list = decoded_list[start_index:end_index]
        model.clear_early_exit_layer_indices()
        return trimmed_text, decoded_list

    # Run through whole dataset
    beam_width = 4; beam_diversity_rate = 0.3
    num_beam_groups = beam_width
    print(f"samples started on {beam_width, beam_diversity_rate}")
    for task_id in problems:
        start_time = time.time()
        print(f"Generating task {task_id}")
        for _ in range(num_samples_per_task):
            prompt = problems[task_id]["prompt"]
            model.set_early_exit_layer(None)
            completion, answer_list = generate_one_completion(prompt)
            model.clear_early_exit_layer_indices()
            print(completion)
            for target_token in answer_list:
                token_idx = tokenizer.encode(target_token)[0]
                appearance_idx = np.count_nonzero(all_appearance[token_idx])
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                # print(f"Generating token '{target_token}'")
                for layer in range(1, num_layers):
                    model.set_early_exit_layer(layer)
                    generated_ids = model.generate(
                        input_ids.to('cuda'),
                        eos_token_id=eos_token,
                        pad_token_id=eos_token,
                        use_cache=False,
                        max_new_tokens=1,
                        num_beam_groups=4,
                        diversity_penalty=0.3,
                        num_beams=4,
                    )
                    model.clear_early_exit_layer_indices()
                    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
                    # print(f"layer {layer+1}: '{trimmed_text}'")
                    if trimmed_text == target_token:
                        print(f"{target_token}: {layer+1}")
                        all_appearance[token_idx][appearance_idx] = layer + 1
                        break
                prompt = prompt + target_token
        end_time = time.time()
        print(f"Task {task_id} took {end_time - start_time} seconds")
        np.save(f"stats_{model_name}.npy", all_appearance)
        print("npy file saved")



if __name__== "__main__":
    main(FLAGS)
