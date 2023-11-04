import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("penguins")[["bill_length_mm", "species", "sex"]]

# Use seaborn factorpot to plot multiple barplots of "bill_length_mm" over "sex" and separate into different subplot columns by "species"
# Do not share y axis across subplots
# SOLUTION START
sns.factorplot(
    x="sex", col="species", y="bill_length_mm", data=df, kind="bar", sharey=False
)
# SOLUTION END

os.makedirs('ans', exist_ok=True)
os.makedirs('input', exist_ok=True)
plt.savefig('ans/oracle_plot.png', bbox_inches ='tight')
with open('input/input1.pkl', 'wb') as file:
    # input is already contained in code_context.txt so we dump a dummy input object
    pickle.dump(None, file)
with open('ans/ans1.pkl', 'wb') as file:
    # test does not require a reference solution object so we dump a dummpy ans object
    pickle.dump(None, file) 
