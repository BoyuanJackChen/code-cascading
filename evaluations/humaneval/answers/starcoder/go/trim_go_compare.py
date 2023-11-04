# %%
import json
import pandas as pd
import os
import subprocess
with open('clean.json', 'r') as f:
    data = json.load(f)


# %%
count = 0
result = []
stop = 0
keys = ['math', 'fmt', 'reflect', 'strings.', 'strconv', 'sort', 'heap', 'rand',  'vector', 'bytes', 'bufio', 'unicode', 'regexp','md5', 'planets']

for i in range(0, len(data)):
    empty = 0
    head_main = 'package main \n \nimport ( '
    head_test = 'package main \n \nimport ( "testing"\n         "github.com/stretchr/testify/assert"\n'
    end = '\n\nfunc main() {}\n'
    
    for j in keys:
        if j in data[i]['answer']:
            head_main += '         "' + j.split('.')[0] + '" \n '
            empty = 1
        if j in data[i]['test']:
            head_test += '         "' + j.split('.')[0] + '" \n '
    if 'big.' in data[i]['answer']:
        head_main += '         "' + 'math/big' + '" \n '
        empty = 1
    if 'big.' in data[i]['test']:
        head_test += '         "' + 'math/big' + '" \n '
    if empty == 0:
        head_main = 'package main \n \n'
    else:
       head_main += ')\n\n'        
    head_test += ')\n\n'
    try:
        function_name =  data[i]['answer'].split(' ')[1].split('(')[0]
    except:
        function_name =  data[i]['answer']

    main =  head_main + data[i]['answer']
    test = head_test + data[i]['test']
    with open("./main.go", 'w') as f:
        f.write(main)
    with open("./main_test.go", 'w') as f:
        f.write(test)
    com = 'go test'

    process = subprocess.Popen(com, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    stdout = stdout.decode().strip()
    stderr = stderr.decode().strip()

    # print(stdout)
    # print(res)
    
    if 'FAIL' in stdout:
        result.append([function_name, 0])
        if 'build failed' in stdout:
            with open("./go_code/main_{}.go".format(i), 'w') as f:
                f.write(main)
            with open("./go_code/main_{}_test.go".format(i), 'w') as f:
                f.write(test)
            count += 1
            print(i)
            #print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")

    elif 'PASS' in stdout:

        result.append([function_name, 1])
    else:
        result.append([function_name, 0])
    # count += 1
    # if count >= 1:
    #     break

out = pd.DataFrame(result, columns=['function_name', 'result'])
out.to_csv('result.csv')
# %%
print(count)
print(sum(out['result']))


# %%
