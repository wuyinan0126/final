# -*- coding: utf-8 -*-
import json
import logging
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

log_format = logging.Formatter('[%(asctime)s]: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(log_format)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console)


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

USE_SERVER = False
SERVER_URL = 'http://10.2.3.83:9126/'

# SERVER_URL = 'http://tqa.23.99.113.200.nip.io:8080/'
all_tags = {
    'Hadoop': [],
    'HDFS': ['NameNode', 'DataNode', 'SecondaryNameNode', '数据块', '名称节点', '数据节点', '第二名称节点', '分布式文件系统', '元数据节点',
             '从元数据节点'],
    'MapReduce': ['Mapper', 'Reducer', 'MapR', 'Map', 'Reduce', 'Shuffle'],
    'YARN': ['JobTracker', 'JobTask', 'ResourceManager', 'NodeManager', '资源管理器', '节点管理器', 'JobHistoryServer']
}


class TqaThread(threading.Thread):
    def __init__(self, question):
        super().__init__()
        self.question = question
        self.user = User.objects.get(pk=1)

    @staticmethod
    def get_tags(question_title, question_description):

        text = (question_title + question_description).lower()
        tag_score = {}
        for tag_k, tag_v in all_tags.items():
            tag_score[tag_k] = 0 if tag_k.lower() not in text else 100
            for key_word in tag_v:
                if key_word.lower() in text:
                    tag_score[tag_k] += 1

        tag_score = sorted(tag_score.items(), key=lambda d: d[1], reverse=True)
        if [tag_score[0][1]] == 0:
            return 'BigData'
        return [tag_score[0][0]]

    def reuse(self, question_title, question_description):
        conn = Sqlite().get_conn()
        cursor = conn.cursor()

        sql = '''
            SELECT q.id, q.title, q.description FROM
            (
                SELECT id, title, description 
                FROM main.qa_question
            ) q
            JOIN 
            (
                SELECT DISTINCT(question_id) AS question_id
                FROM main.qa_answer
                WHERE (positive_votes - negative_votes > 0) OR answer > 0
            ) a
            ON q.id = a.question_id
        '''
        cursor.execute(sql)
        rows = cursor.fetchall()

        questions = [{'id': 0, 'title': question_title, 'desc': question_description}]
        for row in rows:
            id = int(row[0])
            title = row[1]
            description = row[2]
            questions.append({'id': id, 'title': title, 'desc': description})

        payload = {'s': json.dumps(questions)}
        url = SERVER_URL + '?' + urlencode(payload, quote_via=quote_plus)
        result = json.loads(urllib.request.urlopen(url).read().decode('utf-8'))

        reused = ''
        most_similar_id = result['id']
        most_similar_score = float(result['score'])

        if most_similar_id != -1 and most_similar_score != -1:
            sql = '''
                SELECT a.answer_text, q.id, q.title, q.slug FROM
                (
                    SELECT id, title, description, slug
                    FROM main.qa_question 
                    WHERE id = {id} 
                ) q
                JOIN
                (
                    SELECT question_id, answer_text, MAX(positive_votes) 
                    FROM main.qa_answer
                    WHERE question_id = {id} AND (positive_votes - negative_votes > 0)
                ) a
                WHERE q.id = a.question_id
            '''.format(
                id=most_similar_id
            )

            cursor.execute(sql)
            row = cursor.fetchone()
            if row:
                answer_text, question_id, question_title, question_slug = row
                reused = '' if answer_text.strip() == '' else \
                    '导学小助手为您找到论坛中相似的问题：[{similar_question_title}]({link} "{similar_question_title}")，' \
                    '如果解决了您的问题，记得点赞哦～\n\n' \
                    '> {answer_text}\n\n'.format(
                        similar_question_title=question_title,
                        link='/question/%d/%s/' % (question_id, question_slug),
                        answer_text=answer_text.replace('\n\n', '\n\n>')
                    )

        conn.close()
        # logging.info('Reused: ' + reused)
        return reused

    def answer(self, question_title, question_description):
        answer_content_prefix = "导学小助手为您找到了以下相关的资料，如果解决了您的问题，记得点赞哦～\n\n"
        answer_content = ""

        question_title = question_title.replace('@', '(AT)')
        question_description = question_description.replace('@', '(AT)')

        payload = {'q': question_title + "@" + question_description}
        url = SERVER_URL + '?' + urlencode(payload, quote_via=quote_plus)
        results = json.loads(urllib.request.urlopen(url).read().decode('utf-8'))

        for result in results['answers']:
            try:
                result = result[0]
            except:
                result = result
            # if 'course' in result['id']:
            #     filename = os.path.basename(result['id'])
            #     answer_content += '[课程资源（%s）](%s "课程资源")中的相关内容：\n\n' % (filename, result['id'])
            #     answer_content += '> %s\n\n' % result['text']
            # elif 'video' in result['id']:
            #     video_name = os.path.basename(result['id'])
            #     answer_content += '[课程视频（%s）](%s "课程视频")与你的问题相关哦～\n\n' % (video_name, result['id'])
            #     # answer_content += '> %s\n\n' % result['text']
            # el
            if 'wiki' in result['id']:
                url_word = result['id'].split('@')
                url = url_word[0]
                word = url_word[-1] if len(url_word) > 1 else "词条"
                answer_content += '[维基百科（%s）](%s "维基百科")中的相关内容：\n\n' % (word, url)
                answer_content += '> %s\n\n' % result['text']
            elif 'blog' in result['id']:
                url_title = result['id'].split('@')
                url = url_title[1]
                title = url_title[2]
                answer_content += '[博客（%s）](%s "博客")中的相关内容：\n\n' % (title, url)
                answer_content += '> %s\n\n' % result['text']

        return '' if answer_content == '' else answer_content_prefix + answer_content

    def run(self):
        answer = Answer()
        answer.question = self.question
        answer.user = self.user
        question_title = self.question.title
        question_description = self.question.description

        if USE_SERVER:
            answer_content = self.reuse(question_title, question_description)
            if not answer_content:
                answer_content = self.answer(question_title, question_description)

            if answer_content:
                answer.answer_text = answer_content
                answer.save()

# ------------------------------------------------------------------------------
# 自动回答线程 End
# ------------------------------------------------------------------------------
