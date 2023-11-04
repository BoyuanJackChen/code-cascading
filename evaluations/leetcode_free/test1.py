import requests
import json
import time

def get_problem_res(number,answer,slug,cookie):
    print(slug)
    '''
    header_problem = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie' : cookie,
        'Referer': 'https://leetcode.cn/problemset/all/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
    }
    problem_url = f'https://leetcode.cn/problems/{slug}/'
    '''
    with requests.session() as session:
        submit_url = f'https://leetcode.cn/problems/{slug}/submit/'
        cookies = cookie.split(';')
        for c in cookies:
            if c.find('csrftoken') > -1:
                csrf_token = c.split('=')[1]
        headers = {
            'cookie': cookie,
            'Origin': 'https://leetcode.cn',
            'Referer': f'https://leetcode.cn/problems/{slug}/submissions/',
            'Content-Type': 'application/json',
            'X-CSRFToken': csrf_token,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
        }
        headers_check = {
            'Accept': 'application/json, text/plain, */*',
            'Referer': f'https://leetcode.cn/problems/{slug}/submissions/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
            'X-CSRFToken': csrf_token,
            'Host': 'leetcode.cn'
        }
        data = {
            'lang': "python",
            'questionSlug': slug,
            'question_id': number,
            'test_judger': "",
            'test_mode': False,
            'typed_code':answer
        }
        resp = session.post(submit_url,headers=headers,json=data)
        if resp.status_code != 200:
            raise Exception('异常')
        res = resp.json()
        submission_id = res['submission_id']
        MAX_TIME = 20
        times = 0
        while True:
            times += 1
            if times > 20:
                raise Exception('异常')
            check_url = f'https://leetcode.cn/submissions/detail/{submission_id}/check/'
            resp = session.get(check_url,headers=check_url)
            check_data = resp.json()
            if check_data['state'] == 'SUCCESS':
                if check_data['status_code'] == 10:
                    return True
                else:
                    return False
            time.sleep(1)



'''
url = 'https://leetcode.cn/problems/two-sum/submit/'

headers ={
    'cookie':'Hm_lvt_f0faad39bcf8471e3ab3ef70125152c3=1679280149; gr_user_id=4249506b-46d9-4e0c-985d-83dd57202e48; csrftoken=NW8aZhLTrRrPXZMIcNsp3lPdoiB0LzhmBKnte3mDb1W2n0RCN0AACq5RHWiO7I7S; _gid=GA1.2.331901072.1679280152; _bl_uid=7blR8f60gbU8jL12Rmt28wszy2Iw; a2873925c34ecbd2_gr_last_sent_cs1=xi-gua-wang-zai-er; NEW_PROBLEMLIST_PAGE=1; Hm_lpvt_f0faad39bcf8471e3ab3ef70125152c3=1679280367; _ga=GA1.1.148327181.1679280152; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJuZXh0X2FmdGVyX29hdXRoIjoiLyIsIl9hdXRoX3VzZXJfaWQiOiIxMjQ1NzE0IiwiX2F1dGhfdXNlcl9iYWNrZW5kIjoiZGphbmdvLmNvbnRyaWIuYXV0aC5iYWNrZW5kcy5Nb2RlbEJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiIyMWNlM2VmYTg3Yzk5Yzg0NmViZThhOTQ1MzRjMDhlZGQ5NThjNDAyZjg4OWMzMjI0ZTQ4ZGZiMTQxYTkzYWUxIiwiaWQiOjEyNDU3MTQsImVtYWlsIjoiIiwidXNlcm5hbWUiOiJ4aS1ndWEtd2FuZy16YWktZXIiLCJ1c2VyX3NsdWciOiJ4aS1ndWEtd2FuZy16YWktZXIiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jbi9hbGl5dW4tbGMtdXBsb2FkL3VzZXJzL3hpLWd1YS13YW5nLXphaS1lci9hdmF0YXJfMTU4MTMwMjI2My5wbmciLCJwaG9uZV92ZXJpZmllZCI6ZmFsc2UsIl90aW1lc3RhbXAiOjE2NzkyODAzNDkuMDA1NTIzMiwiZXhwaXJlZF90aW1lXyI6MTY4MTg0NDQwMCwidmVyc2lvbl9rZXlfIjowLCJsYXRlc3RfdGltZXN0YW1wXyI6MTY3OTI4MDM2Mn0.f2PKy9Fddm8pmZpE7ryM1mQ4jIPZ6z6XeD2dRLXuWKc; a2873925c34ecbd2_gr_session_id=359d35c5-fd74-40d7-92d8-9238f4970372; a2873925c34ecbd2_gr_last_sent_sid_with_cs1=359d35c5-fd74-40d7-92d8-9238f4970372; a2873925c34ecbd2_gr_session_id_359d35c5-fd74-40d7-92d8-9238f4970372=true; _ga_PDVPZYN3CW=GS1.1.1679285311.2.0.1679285311.0.0.0; a2873925c34ecbd2_gr_cs1=xi-gua-wang-zai-er',
    'Origin': 'https://leetcode.cn',
    'Referer': 'https://leetcode.cn/problems/two-sum/submissions/',
    'Content-Type': 'application/json',
    'X-CSRFToken': 'NW8aZhLTrRrPXZMIcNsp3lPdoiB0LzhmBKnte3mDb1W2n0RCN0AACq5RHWiO7I7S',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
}
data = """
{"question_id":"1","lang":"python","typed_code":"class Solution(object):\n    def twoSum(self, nums, target):\n        \"\"\"\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[int]\n        \"\"\"","test_mode":false,"test_judger":"","questionSlug":"two-sum"}
"""
data = {
'lang': "python",
'questionSlug': "two-sum",
'question_id': "1",
'test_judger': "",
'test_mode': False,
'typed_code': "class Solution(object):\n    def twoSum(self, nums, target):\n        \"\"\"\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[int]\n        \"\"\""
}
resp = requests.post(url,headers=headers,json=data)

print(resp.text)
'''
