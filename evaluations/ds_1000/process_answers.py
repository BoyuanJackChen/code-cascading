import json

filename = "starcoder_k1.json"
outname  = "starcoder.json"
number = 0

for library in ["Matplotlib", "Numpy", "Pandas", "Pytorch", "Scipy", "Sklearn", "Tensorflow"]:
    out_dict = []
    library_number = 0
    input_filename = library + "/" + filename
    output_filename = library + "/" + outname
    # Load filename, which is a json file
    original_data = json.load(open(input_filename))
    for i in range(len(original_data)):
        data_dict = original_data[i]
        prompt = data_dict["prompt"]
        answer = data_dict["answer"]
        output_dict = {"task_id": number, "library_id": library_number, "prompt": prompt, "answer": answer}
        out_dict.append(output_dict)
        number += 1
        library_number += 1
    
    
    with open(output_filename, "w") as o:
        json.dump(out_dict, o, indent=4)

