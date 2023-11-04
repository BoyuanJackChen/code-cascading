import requests
import time


#读取json信息
import json
def read_json(filename):
    with open(filename, 'r') as f:
        data = f.read()
        org_data = json.loads(data)
        return org_data

def write_json(filename,data):
    with open(filename,'w') as f:
        org_data = json.dumps(data)
        f.write(org_data)
#得到网站上每个题的信息
class Util:
    def __init__(self,cookie):
        self.token = ''
        self.cookie = cookie.replace(' ','')
        self.cookie = cookie.split(';')
        cookies = []
        for cookie in self.cookie:
            if cookie.find('csrftoken') >-1:
                self.token = cookie.split('=')[1]
            else:
                cookies.append(cookie)
        self.cookie = ';'.join(cookies)
        self.headers_main = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        }

        self.headers_page = {
            'Accept': '*/*',
            'content-type': 'application/json',
            'Host': 'leetcode.cn',
            'Origin': 'https://leetcode.cn',
            'Referer': 'https://leetcode.cn/problemset/all/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
        }

    #从网站上获取所有题目信息
    def get_web_questions_info(self):
        ret = []
        session = requests.session()
        resp = session.get('https://leetcode.cn/problemset/all/',headers=self.headers_main)
        p_page = 100
        page_idx = 0
        while True:
            print(f'爬取第 {page_idx} 页题目数据,每页100条')
            data = {
                'operationName': "problemsetQuestionList",
                'query': "\n    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {\n  problemsetQuestionList(\n    categorySlug: $categorySlug\n    limit: $limit\n    skip: $skip\n    filters: $filters\n  ) {\n    hasMore\n    total\n    questions {\n      acRate\n      difficulty\n      freqBar\n      frontendQuestionId\n      isFavor\n      paidOnly\n      solutionNum\n      status\n      title\n      titleCn\n      titleSlug\n      topicTags {\n        name\n        nameTranslated\n        id\n        slug\n      }\n      extra {\n        hasVideoSolution\n        topCompanyTags {\n          imgUrl\n          slug\n          numSubscribed\n        }\n      }\n    }\n  }\n}\n",
                'variables' :{
                    'categorySlug': "",
                    'filters': {},
                    'limit': p_page,
                    'skip': page_idx * p_page
                }
            }
            resp = session.post('https://leetcode.cn/graphql/',headers = self.headers_page,json = data)
            #print(resp.cookies)
            page_idx += 1
            data = resp.json()
            if len(data['data']['problemsetQuestionList']['questions']) == 0:
                break
            for q in data['data']['problemsetQuestionList']['questions']:
                insert_data = {
                    'frontendQuestionId' : q['frontendQuestionId'],
                    'titleSlug' : q['titleSlug']
                }
                ret.append(insert_data)
            time.sleep(2)
            #todo
        return ret


    def get_problem_res(self,number,answer,slug):

        with requests.session() as session:
            question_info_url = 'https://leetcode.cn/graphql/'
            submit_url = f'https://leetcode.cn/problems/{slug}/submit/'

            #step1
            headers_question_info = {
                'Cookie': self.cookie + ';csrftoken=' + self.token,
                'Origin': 'https://leetcode.cn',
                'Referer': f'https://leetcode.cn/problems/{slug}/',
                'Content-Type': 'application/json',
                'X-CSRFToken': self.token,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
            }
            question_data = {
                'operationName': "questionData",
                'query': "query questionData($titleSlug: String!) {\n  question(titleSlug: $titleSlug) {\n    questionId\n    questionFrontendId\n    categoryTitle\n    boundTopicId\n    title\n    titleSlug\n    content\n    translatedTitle\n    translatedContent\n    isPaidOnly\n    difficulty\n    likes\n    dislikes\n    isLiked\n    similarQuestions\n    contributors {\n      username\n      profileUrl\n      avatarUrl\n      __typename\n    }\n    langToValidPlayground\n    topicTags {\n      name\n      slug\n      translatedName\n      __typename\n    }\n    companyTagStats\n    codeSnippets {\n      lang\n      langSlug\n      code\n      __typename\n    }\n    stats\n    hints\n    solution {\n      id\n      canSeeDetail\n      __typename\n    }\n    status\n    sampleTestCase\n    metaData\n    judgerAvailable\n    judgeType\n    mysqlSchemas\n    enableRunCode\n    envInfo\n    book {\n      id\n      bookName\n      pressName\n      source\n      shortDescription\n      fullDescription\n      bookImgUrl\n      pressImgUrl\n      productUrl\n      __typename\n    }\n    isSubscribed\n    isDailyQuestion\n    dailyRecordStatus\n    editorType\n    ugcQuestionId\n    style\n    exampleTestcases\n    jsonExampleTestcases\n    __typename\n  }\n}\n",
                'variables': {
                    'titleSlug': slug
                }
            }
            resp = session.post(question_info_url,headers=headers_question_info,json=question_data)
            print('step1:',resp.cookies)
            token = resp.cookies.get('csrftoken')
            if token!= None:
               self.token = token
            question_res = resp.json()
            question_id = question_res['data']['question']['questionId']
            #print('question_id->',question_id)
            question_res = question_res['data']['question']['codeSnippets']

            for q in question_res:
                if q['lang'] == 'Python':
                    code = q['code']
                    break

            data = {
                'lang': "python",
                'questionSlug': slug,
                'question_id': str(question_id),
                'test_judger': "",
                'test_mode': False,
                'typed_code': code + '\n' + answer
            }
            #step2
            while True:
                headers = {
                    'Cookie': self.cookie + ';csrftoken=' + self.token,
                    'Origin': 'https://leetcode.cn',
                    'Referer': f'https://leetcode.cn/problems/{slug}/submissions/',
                    'Content-Type': 'application/json',
                    'X-CSRFToken': self.token,
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
                }

                resp = session.post(submit_url,headers=headers,json=data)
                token = resp.cookies.get('csrftoken')
                print('step2:', resp.cookies)
                if token != None:
                    self.token = token
                if resp.status_code == 403:
                    print(resp.json())
                    print('403')
                    continue
                if resp.status_code == 429:
                    print(resp.status_code,resp.text)
                    raise Exception('异常')
                res = resp.json()
                submission_id = res['submission_id']
                break
            #step3
            headers_check = {
                'Cookie': self.cookie + ';csrftoken=' + self.token,
                'Accept': 'application/json, text/plain, */*',
                'Referer': f'https://leetcode.cn/problems/{slug}/submissions/',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
                'X-CSRFToken': self.token,
                'Host': 'leetcode.cn'
            }
            MAX_TIME = 20
            times = 0
            while True:
                times += 1
                if times > MAX_TIME:
                    raise Exception('异常')
                check_url = f'https://leetcode.cn/submissions/detail/{submission_id}/check/'
                resp = session.get(check_url,headers=headers_check)
                token = resp.cookies.get('csrftoken')
                print('step3:', resp.cookies)
                if token != None:
                    self.token = token
                check_data = resp.json()
                if check_data['state'] == 'SUCCESS':
                    if check_data['status_code'] == 10:
                        return 1
                    else:
                        return 0
                time.sleep(1)

#print(read_json('350M_leetcode_answers_10.json'))
#ret = get_web_questions_info()
#print(len(ret),ret)
