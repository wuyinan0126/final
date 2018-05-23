import json
import threading
import urllib
from urllib.parse import urlencode, quote_plus
import sqlite3

import os
# from tqa_ui.qa.models import Answer
from qa.models import Answer
from django.contrib.auth.models import User

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class Sqlite(object):
    conn = None

    def get_conn(self):
        if self.conn is None:
            self.conn = sqlite3.connect('tqa/tqa.sqlite3')
            self.conn.text_factory = str
        return self.conn


# ------------------------------------------------------------------------------
# 自动回答线程 Begin
# ------------------------------------------------------------------------------

IS_DEBUG = True


class TqaThread(threading.Thread):
    def __init__(self, question):
        super().__init__()
        self.question = question
        self.user = User.objects.get(pk=1)

    def measure(self, question):
        conn = Sqlite().get_conn()
        cursor = conn.cursor()

        sql = '''
            SELECT id, title, description FROM main.qa_question
        '''
        cursor.execute(sql)
        rows = cursor.fetchall()

        for row in rows:
            print(row)

        conn.close()

    def run(self):
        answer = Answer()
        answer.question = self.question
        answer.user = self.user
        question_title = self.question.title
        question_text = self.question.description
        question_content = question_title  # question_title + question_text

        # self.measure(question_content)

        answer_content = "导学小助手为您找到了以下相关的资料，如果解决了您的问题，记得点赞哦～\n\n"
        if IS_DEBUG:
            results = json.loads(
                '{"answers": [['
                '{"score": 0.5, "answer": "答案1", "text": "文本1文本1文本1文本1文本1文本1文本1文本1", "id": "https://zh.wikipedia.org/wiki?curid=1@测试1"},'
                '{"score": 0.4, "answer": "答案2", "text": "文本2文本2文本2文本2文本2文本2文本2文本2", "id": "course/subdir/测试2.pptx"}'
                ']]}', encoding="utf-8")
        else:
            payload = {'q': question_content}
            url = 'http://10.2.3.83:9126/?' + urlencode(payload, quote_via=quote_plus)
            results = json.loads(urllib.request.urlopen(url).read().decode('utf-8'))

        for result in results['answers'][0]:
            if 'wiki' in result['id']:
                url_word = result['id'].split('@')
                url = url_word[0]
                word = url_word[-1] if len(url_word) > 1 else "词条"
                answer_content += '[维基百科（%s）](%s "维基百科")中的相关内容：\n\n' % (word, url)
                answer_content += '> %s\n\n' % result['text']
            else:
                filename = os.path.basename(result['id'])
                answer_content += '[课程资源（%s）](%s "课程资源")中的相关内容：\n\n' % (filename, result['id'])
                answer_content += '> %s\n\n' % result['text']
        if results:
            answer.answer_text = answer_content
            answer.save()

# ------------------------------------------------------------------------------
# 自动回答线程 End
# ------------------------------------------------------------------------------
