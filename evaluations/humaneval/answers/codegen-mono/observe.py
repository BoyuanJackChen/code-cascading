import json
import pandas

with open('16B_k1.json', 'r') as f:
    answer_data_16B = json.load(f)

with open('350M_k1.json', 'r') as f:
    answer_data_350M = json.load(f)

with open('../../data/HumanEval.json', 'r') as f:
    original_data = json.load(f)

# load ../../correctness/codegen.csv into padas dataframe
df = pandas.read_csv('../../correctness/codegen.csv')
print(df.head())
# get all indices where df['350M]==0 and df['16B']==1
all_indices = df[(df['350M']==0) & (df['16B']==1)].index.tolist()
# all_indices = df[(df['16B']==0)].index.tolist()

for i in all_indices:
    prompt = original_data[i]["prompt"]
    answer_16B = answer_data_16B[i]["answer"]
    answer_350M = answer_data_350M[i]["answer"]
    print(i)
    print(prompt)
    print(f"350M: \n{answer_350M}")
    print()
    print(f"16B: \n{answer_16B}")
    input()