# import os
# import time

import sys
import re
import html
import json
import getopt
import requests
import datetime
from requests_toolbelt import MultipartEncoder

turl='https://leetcode.com'
class LEETCode:
    def __init__(self):
        
        self.hea = {
            'referer': turl,
            'user-agent': 'User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;',
        }
        self.cookies = None
        
        self.session = requests.session()
        heads={
            'Cookies':'csrftoken=ijml3IFWgaqB593xio83pkekZMb3qKUHpgMiHZiTXNaW5v11A5zQs9ljfpEGkgPp; gr_user_id=c3ae43bb-672c-4369-a664-3596fffbec76; a2873925c34ecbd2_gr_last_sent_cs1=zou-zou-ting-ting-de-yi-sheng; Hm_lvt_fa218a3ff7179639febdb15e372f411c=1677990345; _ga=GA1.2.134944236.1654932049; __atuvc=0%7C26%2C0%7C27%2C0%7C28%2C3%7C29%2C1%7C30; Hm_lvt_f0faad39bcf8471e3ab3ef70125152c3=1677990311,1677995365; _bl_uid=ULlpLbt3l8RwFggaU1wwpsdshtw2; _ga_PDVPZYN3CW=GS1.1.1678190676.10.1.1678200885.0.0.0; Hm_lpvt_f0faad39bcf8471e3ab3ef70125152c3=1678200888; aliyungf_tc=4cf873a9b7c454194b6cb2fd440b0a990860e91d9d71144fa5328084a285dda2; NEW_PROBLEMLIST_PAGE=1; a2873925c34ecbd2_gr_session_id=a77c8129-8b86-4ab7-b728-faf207ca6d8a; a2873925c34ecbd2_gr_last_sent_sid_with_cs1=a77c8129-8b86-4ab7-b728-faf207ca6d8a; _gid=GA1.2.194745771.1678190678; messages="[[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]\054[\"__json_message\"\0540\05425\054\"\\u60a8\\u5df2\\u7ecf\\u767b\\u51fa\"]]:1pZWvZ:m0T9O6_ClRb7simvLUBeMDaJF4lt7l0yVqTsqLAifYQ"; a2873925c34ecbd2_gr_session_id_a77c8129-8b86-4ab7-b728-faf207ca6d8a=true; _gat=1; _gat_gtag_UA_131851415_1=1; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTY0MjM5OSIsIl9hdXRoX3VzZXJfYmFja2VuZCI6ImF1dGhlbnRpY2F0aW9uLmF1dGhfYmFja2VuZHMuUGhvbmVBdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI4ZDcwZDE2OTg4Y2NjMWVhOTMzNDYzYjc4MjUyMDNiNDJjZWY3MmRjYWJkMTM3NzliNmJhZWYwMDQ4MDA1YWQxIiwiaWQiOjE2NDIzOTksImVtYWlsIjoiY2hpbmFodXdlbmppYUBpY2xvdWQuY29tIiwidXNlcm5hbWUiOiJ6b3Utem91LXRpbmctdGluZy1kZS15aS1zaGVuZyIsInVzZXJfc2x1ZyI6InpvdS16b3UtdGluZy10aW5nLWRlLXlpLXNoZW5nIiwiYXZhdGFyIjoiaHR0cHM6Ly9hc3NldHMubGVldGNvZGUuY24vYWxpeXVuLWxjLXVwbG9hZC91c2Vycy96b3Utem91LXRpbmctdGluZy1kZS15aS1zaGVuZy9hdmF0YXJfMTY2OTcyMDQ5MC5wbmciLCJwaG9uZV92ZXJpZmllZCI6dHJ1ZSwiX3RpbWVzdGFtcCI6MTY3ODIwMDg4Ni40MjQxMDMzLCJleHBpcmVkX3RpbWVfIjoxNjgwNzIxMjAwLCJ2ZXJzaW9uX2tleV8iOjAsImxhdGVzdF90aW1lc3RhbXBfIjoxNjc4MjAwOTA2fQ.Hlv9ycnRxe0IdOMDnfdbCp6QAkdQNCUt8dxxYm6pNaI; a2873925c34ecbd2_gr_cs1=zou-zou-ting-ting-de-yi-sheng'
            }
        self.session.get(url="https://leetcode.com/problemset/all/",headers=heads)
        self.cookies=requests.utils.dict_from_cookiejar(self.session.cookies)
        print(self.cookies.get("csrftoken"))  # 看cookie
        

    def re_get(self):
        #self.get_cookies()
        #self.login()
        self.get_questions()
 

    # 获取初次登陆的cookie
    def get_cookies(self):
        res = requests.get('https://leetcode.com/accounts/login/?next=%2Fproblemset%2Fall%2F',
                           headers=self.hea)
        self.cookies = res.cookies
        print(self.cookies)
        print(self.cookies.get('csrftoken'))
    # 模拟登陆  更换cookies
    def login(self):
        response = MultipartEncoder({
            "csrfmiddlewaretoken": self.cookies.get('csrftoken'),
            "login": "15150651034",
            "password": "qqq123456.",
            "next": "/problems",
        })
        hea = {
            'x-csrftoken': self.cookies.get('csrftoken'),
            'content-type': response.content_type,
            'referer': 'https://leetcode.com/accounts/login/',
            'x-requested-with': 'XMLHttpRequest',
        }
        hea.update(self.hea)
        res = requests.post('https://leetcode.com/accounts/login/',
                            headers=hea,
                            cookies=self.cookies,
                            data=response)
        self.cookies = res.cookies
        # print(res.text)
        pass

    # 获取后台题号和中文标题并获取详细信息
    def get_questions(self):
        hea = {
            'x-csrftoken': self.cookies.get('csrftoken'),
            'content-type': 'application/json',
        }
        all_res = requests.get('https://leetcode.com/api/problems/all/',
                               headers=hea,
                               cookies=self.cookies)
        #print(json.loads(all_res.text)['stat_status_pairs'][::-1])
        buf=json.loads(all_res.text)['stat_status_pairs'][::-1]
        with open('res.json', 'w') as f:
            json.dump(buf, f, indent=4)
      
        level = {1: 'easy', 2: 'medium', 3: 'hard'}
        data = []
        #k=10
        count=0
        for status in json.loads(all_res.text)['stat_status_pairs'][::-1]:
            title = status['stat']['question__title'].replace("'", '"')
           
            print(status['stat']['frontend_question_id'],  title,level[status['difficulty']['level']],status['stat']['question__title_slug'])
            question=self.get_question(status['stat']['question__title_slug'])
            
            if len(question)>0:

                data.append({"number":status['stat']['frontend_question_id'],"title":title,"question":question,"difficulty":status['difficulty']['level']-1})
                
                count+=1
                print("pass:"+str(count))
                #self.write_py(status['stat']['question_id'],title,question)
            else:
                print("No")
           
            #k-=1
            #if k==0:
                #break

        with open('data.json', 'w') as f:
            json.dump(data, f, indent=4)

    # 获取单个题的标题和题
    def get_question(self, slug):
        hea = {
            'x-csrftoken': self.cookies.get('csrftoken'),
            'content-type': 'application/json',
        }
        response = {
            "operationName": "questionData",
            "variables": {
                "titleSlug": ""
            },
            "query": "query questionData($titleSlug: String!) {\n  question(titleSlug: $titleSlug) {\n    "
                     "questionId\n    questionFrontendId\n    boundTopicId\n    title\n    titleSlug\n    content\n   "
                     " translatedTitle\n    translatedContent\n    isPaidOnly\n    difficulty\n    likes\n    "
                     "dislikes\n    isLiked\n    similarQuestions\n    contributors {\n      username\n      "
                     "profileUrl\n      avatarUrl\n      __typename\n    }\n    langToValidPlayground\n    topicTags "
                     "{\n      name\n      slug\n      translatedName\n      __typename\n    }\n    companyTagStats\n "
                     "   codeSnippets {\n      lang\n      langSlug\n      code\n      __typename\n    }\n    stats\n "
                     "   hints\n    solution {\n      id\n      canSeeDetail\n      __typename\n    }\n    status\n   "
                     " sampleTestCase\n    metaData\n    judgerAvailable\n    judgeType\n    mysqlSchemas\n    "
                     "enableRunCode\n    enableTestMode\n    envInfo\n    __typename\n  }\n}\n "
        }

        response['variables']['titleSlug'] = slug
        res = requests.post('https://leetcode.com/graphql',
                            headers=hea,
                            cookies=self.cookies,
                            data=json.dumps(response))
        data = json.loads(res.text)['data']['question']
        temp=""
        if data["content"]:
            
            if len(data["codeSnippets"])<4:
                return temp
            code=str(data["codeSnippets"][3]["code"]).split("\"\"\"")
            codehead=code[0]
            hint=""
            if len(code)>1:
                hint=code[1]
            
            temp+=codehead+"\n"
            html_string="\n"+data["content"].split("<p><strong>Constraints:</strong></p>")[0]
            html_string=html_string.replace("&nbsp;"," ")
           
            stripped_string = re.sub('<[^<]+?>', '', html_string)
            plain_string = html.unescape(stripped_string)
         
            temp+="\t\t\"\"\""+hint+plain_string.replace("\n\n\n", '\n').replace("\n\n", '\n').replace("\n\n", '\n').replace("\n\n", '\n').replace("\n", '\n\t\t')+\
                                 "\n\t\t\"\"\""
            temp=re.sub(r'\n\s*\n', '\n', temp)
            temp = re.sub(r'^\s*#.*\n', '', temp, flags=re.MULTILINE)
            temp = temp.replace("\t", "    ")
            temp+="\n"
   
        return temp
    def write_py(self,tid,title,data):
        with open("source/"+str(tid)+" - "+title+".py", 'w') as file_object:
            file_object.write(data)  
        
   

if __name__ == '__main__':
    lee = LEETCode()
    lee.re_get()
