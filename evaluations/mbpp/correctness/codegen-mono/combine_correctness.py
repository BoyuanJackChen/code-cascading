import pandas as pd

data_idx = pd.read_csv("2B_k1.csv")["question_number"]
data_350M = pd.read_csv("350M_k1.csv")["correctness"]
data_2B   = pd.read_csv("2B_k1.csv")["correctness"]
data_6B   = pd.read_csv("6B_k1.csv")["correctness"]
data_16B  = pd.read_csv("16B_k1.csv")["correctness"]

# Convert all values in each of them to int
data_idx = data_idx.astype(int)
data_350M = data_350M.astype(int)
data_2B   = data_2B.astype(int)
data_6B   = data_6B.astype(int)
data_16B  = data_16B.astype(int)
print(len(data_idx), len(data_350M), len(data_2B), len(data_6B), len(data_16B))

# Concatenate all the above columns into a dataframe
output_df = pd.concat([data_idx, data_350M, data_2B, data_6B, data_16B], axis=1)
output_df = output_df.astype(int)

# Save the dataframe as a csv file
output_df.to_csv("mbpp_d.csv", index=False)