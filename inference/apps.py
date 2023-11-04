from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
import torch
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="Salesforce/codegen-16B-mono", help="Model path")
FLAGS = parser.parse_args()

stop_words = ["\n\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n", "\n\n"]
# 3268 for 350M; 2049 for 2B
all_questions_dict = load_dataset("codeparrot/apps", split="test")
prompt_2049 = ""
prompt_3268 = ""
for question_dict in all_questions_dict:
    if question_dict["problem_id"] == 2049:
        prompt_2049 = question_dict["question"]
    if question_dict["problem_id"] == 3268:
        prompt_3268 = question_dict["question"]

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    trimmed = False
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        answer = answer.lstrip('\n')
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
                    trimmed = True
        answer = answer[:min_i]
        result.append(answer)
    if not trimmed:
        print("This question is not trimmed!")
    return result


def main(args):
    # Initialize model and tokenizer
    checkpoint = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto", eos_token_id=[50256, 628])
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")

    for prompt in [prompt_3268, prompt_2049]:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        start_generating = time.time()
        generated_ids = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=200,
            # attention_mask=torch.zeros(1).cuda(),
            # max_length=200,
            # min_new_tokens=200,
            # do_sample = True,
            # temperature = 1.0,
            # top_k = 0,
            # top_p = 1.0,
            num_beams=4,
            num_beam_groups=4,
            diversity_penalty=0.3,
            # top_k = 20,
            # do_sample = False,
            # penalty_alpha = 0.3
            # stopping_criteria=stopping_criteria,
        )
        print(f"Time to generate is {time.time() - start_generating}")
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)
        generated_len = len(decoded_list) - len(input_ids[0])
        # print(f"generated_ids is {generated_ids[0]}")
        print(f"decoded_list is {decoded_list}")
        # print(f"decoded list has length {len(decoded_list)}; prompt has length {len(input_ids[0])};"
        #       f"generated has length {generated_len}")
        print(f"per token time is {(time.time()-start_generating)/generated_len}")

        # Print text
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt = tokenizer.decode(prompt_ids[0])
        trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
        print(f"\ngenerated_text is:\n{generated_text[0]}")
        # print("trimmed_text is:")
        # print(trimmed_text[0])

if __name__== "__main__":
    main(FLAGS)