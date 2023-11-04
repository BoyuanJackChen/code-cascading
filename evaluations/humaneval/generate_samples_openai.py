from human_eval.data import write_jsonl, read_problems
import openai
openai.api_key = "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"

def generate_one_completion(prompt):
    result = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=["\n\n"]
    )
    return result

problems = read_problems()

num_samples_per_task = 3
select_ids = ['HumanEval/0', 'HumanEval/1']
print(len(problems))
print(type(problems))
print(problems['HumanEval/0'].keys())
print(problems['HumanEval/105']['canonical_solution'])
print(problems['HumanEval/105']['test'])
print("samples started")
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    # for task_id in select_ids
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)