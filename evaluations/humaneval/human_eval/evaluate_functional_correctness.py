import fire
import sys

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 2,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    # problem_file: str = "/Users/boyuanchen/Desktop/human-eval/data/example_problem.jsonl",
):
    # print(f"\n\nHUMAN_EVAL is {HUMAN_EVAL}\n\n")
    # HUMAN_EVAL is /Users/boyuanchen/Desktop/human-eval/human_eval/../data/HumanEval.jsonl.gz
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    print(f"sample file is {sample_file}")
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)


def main():
    fire.Fire(entry_point)


sys.exit(main())
