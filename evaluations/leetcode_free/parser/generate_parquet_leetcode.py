from datasets import load_dataset
import pandas as pd
import pyarrow.parquet as pq

def append_difficulty(row):
    question = row['question']
    difficulty = row['difficulty']
    # Find the last line of the question
    last_line = question.split('\n')[-2]
    # Extract the indentation from the last line
    indentation = last_line[:-3]
    # Append the appropriate string based on difficulty
    if difficulty == 0:
        question = question + indentation + "# Easy\n"
    elif difficulty == 1:
        question = question + indentation + "# Medium\n"
    elif difficulty == 2:
        question = question + indentation + "# Hard\n"
    return question

def get_stats(df):
    difficulty_stats = df.groupby("difficulty").count()
    print(difficulty_stats)

df = pd.read_json('data.json')
df = df.reindex(columns=["number", "difficulty", "question"])
get_stats(df)
print(df.iloc[731]['question'])
df.to_parquet('leetcode_free_questions_labeled.parquet')