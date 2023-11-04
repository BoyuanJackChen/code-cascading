from ds1000 import DS1000Dataset

ds_data = DS1000Dataset("ds1000_data") # loads all questions into RAM

# DS1000Dataset is splitted by different libraries
ds_data["Pandas"] # list of 291 Pandas questions
ds_data["Pandas"][0] # get first Pandas question
# print(ds_data)
# input()
# print(ds_data["Pandas"])
# input()
# print(ds_data["Pandas"][0])
# input()
# print(ds_data["Pandas"][0].keys())
# input()
print(ds_data["Pandas"][1]["reference_code"])
input()
for key in ds_data["Pandas"][0].keys():
    print(f"{key}:\n{ds_data['Pandas'][0][key]}\n\n")
    input()

def process_prompt(prompt):
    prompt = prompt.replace("\n\n\n", "\n")
    prompt = prompt.replace("\n\n", "\n")
    # prompt cut off the first line.
    prompt = prompt[prompt.find("\n") + 1:]
    prompt = "Complete the python program given the prompt below:\n" + prompt
    # Get the good half-line
    preserved_prompt = prompt[prompt.rfind("</code>"):]
    preserved_prompt = preserved_prompt[preserved_prompt.find("\n") + 1:]
    preserved_prompt = preserved_prompt[:preserved_prompt.find("\n")]
    # Trim the prompt off from the last occurrence of "</code>"
    prompt = prompt[:prompt.find("</code>")]
    # replace "A:\n<code>" with "\nCode:"
    prompt = prompt.replace("A:\n<code>", "\nCode:")
    # prompt += "\n" + preserved_prompt
    return prompt

prompt = process_prompt(ds_data["Pandas"][0]["prompt"])
print(prompt)
input()

# look at reference_code to understand the question
reference_code = ds_data["Pandas"][156]["reference_code"]

# run official test on the generated code to
# is_correct = ds_data["Pandas"][0].test(generated_code)

print(ds_data["Pandas"][156]['prompt'])

# checkout other available information in each question
# we will go over them in the next section
# print(ds_data["Pandas"][0].keys())
# print(ds_data["Pandas"][0]["perturbation_origin_id"])
# print(ds_data["Numpy"][0]["perturbation_origin_id"])
for key in ds_data["Numpy"][0].keys():
    print(key)
    # print(ds_data["Numpy"][0][key])

# for question in ds_data["Matplotlib"]:
#     print(question.keys())

the_sum = 0
for key in ["Matplotlib", "Numpy", "Pandas", "Pytorch", "Scipy", "Sklearn", "Tensorflow"]:
    dataset = ds_data[key]
    print(len(dataset))
    the_sum += len(dataset)
print(the_sum)