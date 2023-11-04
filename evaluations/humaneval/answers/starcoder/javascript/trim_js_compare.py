# %%
import json
import execjs
import pandas as pd
with open('clean.json', 'r') as f:
    data = json.load(f)


# %%
count = 0
result = []
stop = 0
for i in range(0, len(data)):
    code = data[i]['answer']
    if code == '':
        try:
            function_name = data[i]['prompt'].split('>>>')[1].split('(')[0]
            result.append([function_name, 0])
        except:
            try:
                function_name = data[i]['prompt'].split(':\n')[1].split('(')[0]
                result.append([function_name, 0])
            except:
                print(data[i]['prompt'])
        continue
    function_name = code.split(' ')[1]
    asserts = data[i]['test'].split('console.assert')
    # print(data[i]['test'])
    res = 1

    for j in range(1, len(asserts)):
        comlist = asserts[j]
        if j == len(asserts) - 1:
            for k in range(len(comlist) - 1, 0, -1):
                if comlist[k] == '}':
                    comlist = comlist[:k]
                    # print(comlist)
                    break
        # print(comlist)
        ass = '\n\nconst out = () => {\n  return ' + comlist + ';  \n};\n '  
        context = execjs.compile(code + ass)
        # print(code + ass)
        # print(context.call('out'))
        try:
            if not context.call('out'):
                res = 0
        except Exception as e:
            # print(i)
            # print(code + ass)
            # print(data[i]['answer'])
            # print(data[i]['test'])
            # print(e)
            # print(j,len(asserts))
            res = 0
            count += 1
            # print(context.call('out'))
            break
        # if truth != context.call('out'):
        #     res = 0
        #     break

        # print(context.call('out'))  
    
    result.append([function_name, res])  
    # print(result)
    if stop == 1:
        break
    # count += 1
    # if count >= 1:
    #     break

out = pd.DataFrame(result, columns=['function_name', 'result'])
out.to_csv('result.csv')
# %%
print(count)
print(sum(out['result']))

# %%

# i = 0
# code = data[0]['answer']
# print(code)
# # %%
# code = data[0]['answer']
# print(data[0]['test'].split(' ')[1][4:])

# # %%
func = '(hasCloseElements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) === true)'
ass = '\n\nconst out = () => {\n  return ' + func + ';  \n};\n '    
code+= ass
print(code)
# %%
context = execjs.compile(code)
print(context.call('out'))
# # %%
# listl = data[0]['test'].split('assert')
# for i in range(1, len(listl)):
#     print(listl[i].split('(')[2])
#     print(listl[i].split('(')[2].split(')')[0])
#     if 'true' in listl[i].split('(')[2]:
#         print(1)
# %%
