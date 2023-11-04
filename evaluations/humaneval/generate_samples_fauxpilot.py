from human_eval.data import write_jsonl, read_problems
import openai
openai.api_key = 'dummy'
openai.api_base = 'http://0.0.0.0:5000/v1'

beam_width = 8
all_diversity_rate = [0.0, 0.3, 0.5, 0.7, 1,0, 2.0]
for beam_diversity_rate in all_diversity_rate:
# beam_diversity_rate = 2.0
    def generate_one_completion(prompt):
        result = openai.Completion.create(
            engine='codegen',
            prompt=prompt,
            max_tokens=200,
            stop=["\n\n"],
            temperature=0.7,
            beam_width=beam_width,
            beam_search_diversity_rate=beam_diversity_rate
        )
        return result

    problems = read_problems()

    num_samples_per_task = 3
    # select_ids = ['HumanEval/0', 'HumanEval/1']
    # print(len(problems))
    # print(type(problems))
    # print(problems['HumanEval/0'].keys())
    # print(problems['HumanEval/105']['canonical_solution'])
    # print(problems['HumanEval/105']['test'])
    print("samples started")
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
        # for task_id in select_ids
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    write_jsonl(f"samples_{beam_width}_{beam_diversity_rate}.jsonl", samples)
