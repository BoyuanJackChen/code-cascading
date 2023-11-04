import pandas as pd
import numpy as np


def g(df):
    df.loc[df["name"].str.split().str.len() == 2, "2_name"] = (
        df["name"].str.split().str[-1]
    )
    df.loc[df["name"].str.split().str.len() == 2, "name"] = (
        df["name"].str.split().str[0]
    )
    df.rename(columns={"name": "1_name"}, inplace=True)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {"name": ["Jack Fine", "Kim Q. Danger", "Jane Smith", "Zhongli"]}
        )

    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    result = g(df)
    print(result)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
