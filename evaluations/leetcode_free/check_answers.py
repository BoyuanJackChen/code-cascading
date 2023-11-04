import util
import time
import csv
import traceback
import os
import pandas as pd
import sys

cookie = 'NEW_PROBLEMLIST_PAGE=1; gr_user_id=82bf8c14-504a-410d-9725-85c726400c07; Hm_lvt_f0faad39bcf8471e3ab3ef70125152c3=1681752933; _bl_uid=2LlymgpOl1m4C0awUnsdrU8ia4Cp; _gid=GA1.2.971910330.1682970400; gioCookie=yes; messages="[[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]]:1puN4L:knaubJpQHkkh9iR7CbTvEE3acaKhCBQDSU9HWq0A1wA"; _gat_gtag_UA_131851415_1=1; csrftoken=UEAl690YZrt8dgdFK7R3HF7nkVVVpVFBrXfWod8hmF5SvhfT0QJ1nLoWOMIRUkEO; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiNDUxMjgwMyIsIl9hdXRoX3VzZXJfYmFja2VuZCI6ImF1dGhlbnRpY2F0aW9uLmF1dGhfYmFja2VuZHMuUGhvbmVBdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiJiNjY3ODdiYTdlYjRiZmJjMDFlNzA4Njg5ODZmMzVkYThiMzljODNhNzZiYTQ1N2VkZDFiNTk2NTc2OWZiZThmIiwiaWQiOjQ1MTI4MDMsImVtYWlsIjoiZ2V3ZWlzaGljOTgyOTJAMTYzLmNvbSIsInVzZXJuYW1lIjoiYWRtaXJpbmctN2FtcG9ydGl6aiIsInVzZXJfc2x1ZyI6ImFkbWlyaW5nLTdhbXBvcnRpemoiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jbi9hbGl5dW4tbGMtdXBsb2FkL2RlZmF1bHRfYXZhdGFyLnBuZyIsInBob25lX3ZlcmlmaWVkIjp0cnVlLCJfdGltZXN0YW1wIjoxNjgzMTYxMDYyLjQ2NzEzMTksImV4cGlyZWRfdGltZV8iOjE2ODU3MzI0MDAsInZlcnNpb25fa2V5XyI6MX0.8CXxNf4pWVtlUUh1ka49S3GMwL1w_UvUVp_ReU9HnMk; a2873925c34ecbd2_gr_last_sent_sid_with_cs1=276dd7ee-53e1-474e-9ada-aaa37615b40c; a2873925c34ecbd2_gr_last_sent_cs1=admiring-7amportizj; a2873925c34ecbd2_gr_session_id=276dd7ee-53e1-474e-9ada-aaa37615b40c; a2873925c34ecbd2_gr_session_id_sent_vst=276dd7ee-53e1-474e-9ada-aaa37615b40c; _ga_PDVPZYN3CW=GS1.1.1683159338.31.1.1683161067.0.0.0; NEW_QUESTION_DETAIL_PAGE_V2=1; a2873925c34ecbd2_gr_cs1=admiring-7amportizj; Hm_lpvt_f0faad39bcf8471e3ab3ef70125152c3=1683161069; _ga=GA1.2.1037603726.1681752932; _gat=1'
json_file_name = './answers/16B_k1.json'
output_file_name = './correctness/16B_k1.csv'

time_delay = 15
_util = util.Util(cookie)

try:
    json_data = util.read_json(json_file_name)
    print(json_data)
    print(f'读取json数据数量为: {len(json_data)}')
except:
    print('读取json文件错误')

try:
    problem_data = util.read_json('questions.txt')
    print(problem_data)
except:
    traceback.print_exc()
    print('获取网站题目信息失败')

# Read existing data
# json_file_name.replace('.json', '.csv')
output_file_exists = os.path.exists(output_file_name)
all_problems = []
existing_problems = []
if output_file_exists:
    with open(output_file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] != 'question_number':
                existing_problems.append(int(row[0]))
for j_data in json_data:
    number = j_data['number']
    if number not in all_problems and number not in existing_problems:
        all_problems.append(number)

# If dong missing files, this is it.
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
                    sys.exit()
                if not is_excpt:
                    writer.writerow([number, ret])
                    f.flush()
                    break
                trytime += 1
                time.sleep(time_delay)
            time.sleep(time_delay)

# Read and sort the file and rewrite with pandas
data = pd.read_csv(output_file_name)
data = data.sort_values(by=['question_number'])
data.to_csv(output_file_name, index=False)