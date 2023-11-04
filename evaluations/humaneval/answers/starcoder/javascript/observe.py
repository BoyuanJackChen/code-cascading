import pandas as pd

# read in result.csv
df = pd.read_csv('result.csv')
# count 1's in df['result']
print(df['result'].value_counts())
