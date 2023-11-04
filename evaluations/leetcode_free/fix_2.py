import util
import time
import csv
import traceback
import os
import pandas as pd

cookie = 'gr_user_id=e30ede61-879a-43fd-9cc8-90b1f72a9f49; _bl_uid=XRlg8f2FeIde6dgg82w8umy4C9FL; NEW_PROBLEMLIST_PAGE=1; _gid=GA1.2.770995186.1681322576; Hm_lvt_f0faad39bcf8471e3ab3ef70125152c3=1679169966,1681322577; NEW_QUESTION_DETAIL_PAGE_V2=1; _gat=1; messages="[[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]]:1pmibv:YIZB1avBNUqw2JVcS7qGOW9qpHQ0DIv6mWZdN68TYRs"; _gat_gtag_UA_131851415_1=1; csrftoken=kgqk7fTCTSi1auXnlQJSL28SZOScgv65pnsERP6rvfOnA0Ivt54YRM7nWsUmPi7m; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiNDUxMjc2NyIsIl9hdXRoX3VzZXJfYmFja2VuZCI6ImF1dGhlbnRpY2F0aW9uLmF1dGhfYmFja2VuZHMuUGhvbmVBdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiJhMjBhM2Y3ZWI4Y2ZhNmQyZjYwOGU4NTllM2ExODQ4ZmJiYTFiYzhkYTg5ZTdlN2UxOTU3YWNhZTliMzZjYjczIiwiaWQiOjQ1MTI3NjcsImVtYWlsIjoibGlhb2t1YW5vMjUyNzUwQDE2My5jb20iLCJ1c2VybmFtZSI6ImJvcmluZy1ob2ZzdGFkdGVyMHRxIiwidXNlcl9zbHVnIjoiYm9yaW5nLWhvZnN0YWR0ZXIwdHEiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jbi9hbGl5dW4tbGMtdXBsb2FkL2RlZmF1bHRfYXZhdGFyLnBuZyIsInBob25lX3ZlcmlmaWVkIjp0cnVlLCJfdGltZXN0YW1wIjoxNjgxMzM3MjE3Ljk3NzIwMzQsImV4cGlyZWRfdGltZV8iOjE2ODM5MTgwMDAsInZlcnNpb25fa2V5XyI6MX0.qglB5XYezumIaU7B28ACmrRDzk5k2HgWsmuw4rD-DsM; a2873925c34ecbd2_gr_last_sent_sid_with_cs1=c9d09595-6219-434f-9e5b-ca64a237a0bc; a2873925c34ecbd2_gr_last_sent_cs1=boring-hofstadter0tq; a2873925c34ecbd2_gr_session_id=c9d09595-6219-434f-9e5b-ca64a237a0bc; a2873925c34ecbd2_gr_session_id_c9d09595-6219-434f-9e5b-ca64a237a0bc=true; _ga_PDVPZYN3CW=GS1.1.1681337201.3.1.1681337227.0.0.0; Hm_lpvt_f0faad39bcf8471e3ab3ef70125152c3=1681337229; _ga=GA1.2.659908265.1679169965; a2873925c34ecbd2_gr_cs1=boring-hofstadter0tq'
cookie = 'gr_user_id=e30ede61-879a-43fd-9cc8-90b1f72a9f49; _bl_uid=XRlg8f2FeIde6dgg82w8umy4C9FL; NEW_PROBLEMLIST_PAGE=1; _gid=GA1.2.770995186.1681322576; Hm_lvt_f0faad39bcf8471e3ab3ef70125152c3=1679169966,1681322577; NEW_QUESTION_DETAIL_PAGE_V2=1; messages="[[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]]:1pmkyP:y0nqHqY0PGABqgJW18jF_ZXTAXaL2pVrsUwwH2ne1Po"; _gat_gtag_UA_131851415_1=1; csrftoken=NV9lG9BHdnGq8w1nGta9sHkM55SA0csLucqpr5QIBXSMADVYgVpAfLaU2XdoDrAi; _ga=GA1.1.659908265.1679169965; _ga_PDVPZYN3CW=GS1.1.1681346281.4.1.1681346292.0.0.0; Hm_lpvt_f0faad39bcf8471e3ab3ef70125152c3=1681346293; a2873925c34ecbd2_gr_last_sent_sid_with_cs1=c2b8c8d3-1321-41a2-ae1a-5034e510c01e; a2873925c34ecbd2_gr_last_sent_cs1=abcs1; a2873925c34ecbd2_gr_cs1=abcs1; a2873925c34ecbd2_gr_session_id=c2b8c8d3-1321-41a2-ae1a-5034e510c01e; a2873925c34ecbd2_gr_session_id_c2b8c8d3-1321-41a2-ae1a-5034e510c01e=true; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMzU4NTY2OSIsIl9hdXRoX3VzZXJfYmFja2VuZCI6ImF1dGhlbnRpY2F0aW9uLmF1dGhfYmFja2VuZHMuUGhvbmVBdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI3Zjk3ZTI5OTNjMzZjMTRjNDJkOTc2MDExNTcyNTY2YzExM2U1YjhlOTJhYThlNTg5ZTU5MzViM2I3ODk0MDNlIiwiaWQiOjM1ODU2NjksImVtYWlsIjoienh5YmVpamluZ18yMDIwQDE2My5jb20iLCJ1c2VybmFtZSI6ImFiY3MxIiwidXNlcl9zbHVnIjoiYWJjczEiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jbi9hbGl5dW4tbGMtdXBsb2FkL2RlZmF1bHRfYXZhdGFyLnBuZyIsInBob25lX3ZlcmlmaWVkIjp0cnVlLCJfdGltZXN0YW1wIjoxNjgxMzQ2MjkyLjg2ODE2NSwiZXhwaXJlZF90aW1lXyI6MTY4MzkxODAwMCwidmVyc2lvbl9rZXlfIjoxNiwibGF0ZXN0X3RpbWVzdGFtcF8iOjE2ODEzNDYyOTl9.Hobag1BxLb8SBuRp6cWFVzhTxFB8bdg9FHnBZ0nRRic'
cookie = 'gr_user_id=e30ede61-879a-43fd-9cc8-90b1f72a9f49; _bl_uid=XRlg8f2FeIde6dgg82w8umy4C9FL; NEW_PROBLEMLIST_PAGE=1; _gid=GA1.2.770995186.1681322576; Hm_lvt_f0faad39bcf8471e3ab3ef70125152c3=1679169966,1681322577; messages="[[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]]:1pmlCF:nBbITOIcGkj5TclUGXAyOpj87KzlQpMuHDI5p1RtcR8"; csrftoken=PmEFjRnjHHW0YlfPmvZriDSVyZGXGWGzt3RkfJil4lxbIX8nENt06mG29HeqxqC9; a2873925c34ecbd2_gr_last_sent_cs1=thirsty-karetuo; _ga_PDVPZYN3CW=GS1.1.1681346281.4.1.1681347155.0.0.0; _ga=GA1.2.659908265.1679169965; _gat=1; a2873925c34ecbd2_gr_session_id=8a41c1d5-693b-42ca-abef-947f18180e56; a2873925c34ecbd2_gr_last_sent_sid_with_cs1=8a41c1d5-693b-42ca-abef-947f18180e56; a2873925c34ecbd2_gr_cs1=thirsty-karetuo; a2873925c34ecbd2_gr_session_id_8a41c1d5-693b-42ca-abef-947f18180e56=true; Hm_lpvt_f0faad39bcf8471e3ab3ef70125152c3=1681353057'
json_file_name = './answers/2B_k1.json'
correct_file_name = './correctness/2B_k1.csv'

time_delay = 15

_util = util.Util(cookie)

try:
    json_data = util.read_json(json_file_name)
    # print(json_data)
    print(f'读取json数据数量为: {len(json_data)}')
except:
    print('读取json文件错误')

try:
    problem_data = util.read_json('questions.txt')
    # print(problem_data)
except:
    traceback.print_exc()
    print('获取网站题目信息失败')

# Find bad problems
output_file_name = correct_file_name
output_file_exists = os.path.exists(output_file_name)
bad_problems = []
with open(correct_file_name, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row)>=2 and row[1] == '2':
            bad_problems.append(int(row[0]))

# Find missing
target_file = "./correctness/2B_k1.csv"
paradigm_file = "./correctness/350M_k1.csv"
data = pd.read_csv(target_file)
paradigm_data = pd.read_csv(paradigm_file)
missing_numbers = []
for number in paradigm_data['question_number']:
    if number not in data['question_number']:
        missing_numbers.append(number)

all_problems = bad_problems + missing_numbers
print(all_problems)

with open(output_file_name, 'a+', newline='') as f:
    writer = csv.writer(f)
    if not output_file_exists:
        writer.writerow(['question_number','correctness'])
    for j_data in json_data:
        number = j_data['number']
        if number not in all_problems:
            continue
        answer = j_data['answer']
        if str(number) in problem_data.keys():
            slug = problem_data[str(number)]
            print(f'正在处理: number - {number} {slug} 题目')
            trytime = 0
            is_excpt = False
            while True:
                if trytime > 2:
                    writer.writerow([number, 2])
                    f.flush()
                    break
                try:
                    ret = _util.get_problem_res(str(number),answer,slug)
                except Exception as e:
                    is_excpt = True
                    traceback.print_exc()
                if is_excpt == False:
                    writer.writerow([number, ret])
                    f.flush()
                    break
                trytime += 1
                time.sleep(time_delay)
            time.sleep(time_delay)