import numpy as np
from transformers import AutoTokenizer


def get_on_difficulty_idx(all_appearance, tokenizer, start_idx, end_idx):
    # calculate average of non-zero values in second dimension of the array
    masked = np.ma.masked_equal(all_appearance, 0)
    average = masked.mean(axis=1)
    print(average.shape)
    # create a new array to store the average and original row indices
    new_arr = np.zeros((all_appearance.shape[0], 2))
    print(new_arr.shape)
    for i in range(all_appearance.shape[0]):
        new_arr[i][0] = average[i]
        new_arr[i][1] = i
    # rank the new array only based on the first column
    ranked_idx = np.argsort(new_arr[:, 0])
    # print the row indices of rows ranked between start_idx and end_idx
    for i in range(start_idx, end_idx + 1):
        token_idx = new_arr[ranked_idx][i][1]
        average = new_arr[ranked_idx][i][0]
        token_str = tokenizer.decode(int(token_idx))
        print(f"{token_str}: {average}")


if __name__ == "__main__":
    loaded = 'Salesforce/codegen-16B-mono'
    all_appearance = np.load("stats_codegen-16B-mono.npy")
    tokenizer = AutoTokenizer.from_pretrained(loaded, device_map="auto")
    get_on_difficulty_idx(all_appearance, tokenizer, 0, 1000)