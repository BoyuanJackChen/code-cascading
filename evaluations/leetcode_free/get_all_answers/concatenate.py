import json

# 16B
data = []
data_files = ["16B_k1_start0_end100.json", "16B_k1_start100_end200.json", "16B_k1_start200_end300.json",
              "16B_k1_start300_end400.json", "16B_k1_start400_end500.json", "16B_k1_start500_end600.json",
              "16B_k1_start600_end700.json", "16B_k1_start700_end800.json", "16B_k1_start800_end900.json",
              "16B_k1_start900_end1000.json", "16B_k1_start1000_end1100.json",
              "16B_k1_start1100_end1200.json", "16B_k1_start1200_end1300.json", "16B_k1_start1300_end1400.json",
              "16B_k1_start1400_end1500.json", "16B_k1_start1500_end1600.json", "16B_k1_start1600_end1700.json",
              "16B_k1_start1700_end1800.json", "16B_k1_start1800_end1900.json", "16B_k1_start1900_end2000.json",
              "16B_k1_start2000_end2100.json", "16B_k1_start2100_end2200.json", "16B_k1_start2200_end2300.json",
              "16B_k1_start2300_end2400.json", "16B_k1_start2400_end2500.json", "16B_k1_start2500_end2600.json"]

# 6B
# data = []
# data_files = ["6B_k1_start0_end200.json", "6B_k1_start200_end400.json", "6B_k1_start400_end600.json",
#               "6B_k1_start600_end800.json", "6B_k1_start800_end1000.json", "6B_k1_start1000_end1200.json",
#               "6B_k1_start1200_end1400.json", "6B_k1_start1400_end1600.json", "6B_k1_start1600_end1800.json",
#               "6B_k1_start1800_end2000.json", "6B_k1_start2000_end2200.json", "6B_k1_start2200_end2400.json",
#               "6B_k1_start2400_end2600.json"]

# 2B
# data = []
# data_files = ["2B_k1_start0_end400.json", "2B_k1_start400_end800.json", "2B_k1_start800_end1200.json",
#               "2B_k1_start1200_end1600.json", "2B_k1_start1600_end2000.json", "2B_k1_start2000_end2600.json"]

# 350M
# data = []
# data_files = ["350M_k1_start0_end800.json", "350M_k1_start800_end1600.json", "350M_k1_start1600_end2600.json"]

print(f"Original length: {len(data)}")

for file in data_files:
    with open(file, 'r') as f:
        file_data = json.load(f)
        print(f"{file} length: {len(file_data)}")
        data.extend(file_data)

unique_data = [dict(t) for t in set(tuple(d.items()) for d in data)]
print(len(unique_data))

with open('16B_k1.json', 'w') as f:
    json.dump(unique_data, f, indent=4)