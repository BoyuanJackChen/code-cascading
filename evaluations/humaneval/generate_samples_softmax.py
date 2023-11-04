from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CodeGenModel
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--loaded_model', type=str, default='Salesforce/codegen-16B-mono')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_samples_per_task', type=int, default=1)
parser.add_argument('--beam_width', type=int, default=4)
parser.add_argument('--num_beam_groups', type=int, default=4)
parser.add_argument('--beam_diversity_rate', type=float, default=0.3)
parser.add_argument('--early_exit_layer', type=int, default=None)
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
    num_samples_per_task = args.num_samples_per_task
    tokenizer = AutoTokenizer.from_pretrained(loaded, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(loaded, device_map="auto")
    if args.early_exit_layer is not None:
        model.set_early_exit_layer(args.early_exit_layer)

    beam_width = args.beam_width
    num_beam_groups = args.num_beam_groups
    beam_diversity_rate = args.beam_diversity_rate
    all_beam_widths = [4]
    all_diversity = [0.3]
    softmax_thresholds = [0.95, 0.9, 0.8]
    all_exit_layers = np.array([])

    def generate_one_completion(prompt, all_exit_layers):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(
            input_ids.to('cuda'),
            max_new_tokens=200,
            eos_token_id=eos_token,
            pad_token_id=eos_token,
            num_beams=beam_width,
            num_beam_groups=num_beam_groups,
            diversity_penalty=beam_diversity_rate,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        early_layers = model.get_early_exit_layer_indices()
        early_layers = early_layers[:early_layers.count_nonzero()].numpy()
        all_exit_layers = np.append(all_exit_layers, early_layers)
        # print(f"generated_text is:\n{generated_text}")
        trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
        print(f"trimmed_text is:\n{trimmed_text}")
        model.clear_early_exit_layer_indices()
        return trimmed_text, all_exit_layers

    for softmax_threshold in softmax_thresholds:
        model.set_softmax_threshold(softmax_threshold)
        for beam_width in all_beam_widths:
            for beam_diversity_rate in all_diversity:
                num_beam_groups = beam_width
                print(f"samples started on {beam_width, beam_diversity_rate, softmax_threshold}")
                samples = []
                for task_id in problems:
                    print(f"Generating task {task_id}")
                    for _ in range(num_samples_per_task):
                        completion, all_exit_layers = generate_one_completion(problems[task_id]["prompt"], all_exit_layers)
                        sample = dict(
                            task_id=task_id,
                            completion=completion
                        )
                        samples.append(sample)
                        print(all_exit_layers)

                filename = f"softmax_{softmax_threshold}_{beam_width}_{num_beam_groups}_{beam_diversity_rate}_transformers{model_name}"
                if args.early_exit_layer is not None:
                    filename += f"_early{args.early_exit_layer}"
                write_jsonl(filename + ".jsonl", samples)
                np.save(filename + "_early_layers.npy", all_exit_layers)
                print(np.mean(all_exit_layers))

    # print("samples started")
    # samples = []
    # for task_id in problems:
    #     print(f"Generating task {task_id}")
    #     for _ in range(num_samples_per_task):
    #         sample = dict(
    #             task_id=task_id,
    #             completion=generate_one_completion(problems[task_id]["prompt"])
    #         )
    #         samples.append(sample)
    #
    # filename = f"samples_{beam_width}_{num_beam_groups}_{beam_diversity_rate}_transformers{model_name}"
    # if args.early_exit_layer is not None:
    #     filename += f"_early{args.early_exit_layer}"
    # write_jsonl(filename+".jsonl", samples)


if __name__== "__main__":
    main(FLAGS)
