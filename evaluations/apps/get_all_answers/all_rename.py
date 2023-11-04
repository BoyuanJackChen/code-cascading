import json
from datasets import load_dataset
import os

all_data = []
for start in range(1200, 5000, 100):
    end = start + 100
    output_file_name = f"6B/6B_k1_start{start}_end{end}.json"
    # Rename the file
    os.rename(output_file_name, f"6B/start{start}_end{end-1}.json")