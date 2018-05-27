-- session
-- INSERT OR REPLACE INTO main.hitcount_hit (created, ip, user_agent, hitcount_id, user_id, session) VALUES (
--   datetime('now'), '127.0.0.1',
--   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36',
--   1, NULL, 'qio38txk4r22whmqa75yjie79d7m6nsz'
-- );
-- tag: 大数据
INSERT OR REPLACE INTO main.taggit_tag (id, name, slug) VALUES (
  1, '大数据', '大数据'
);

-- question
INSERT OR REPLACE INTO main.qa_question (id, description, pub_date, reward, closed, user_id, negative_votes, positive_votes, total_points, slug, title)
VALUES (
  1,
  '是只有存放数据的那些节点参与，还是数据会传送到所有节点上，所有节点都参与？',
  datetime('now'), 0, 0, 2, 0, 0, 0,
  'mapreduce过程中有哪些节点会参与', 'mapreduce过程中有哪些节点会参与？'
);
-- tag
INSERT OR REPLACE INTO main.taggit_taggeditem (object_id, content_type_id, tag_id) VALUES (
  1, 13, 1
);
-- answer
INSERT OR REPLACE INTO main.qa_answer (answer_text, pub_date, question_id, user_id, answer, negative_votes, positive_votes, total_points, updated)
VALUES (
  '由Master节点将M个Map任务和R个Reduce任务分配到空闲的节点上运行。输入文件被分成固定大小 （默认为64 MB， 用户可以调整） 的M个分片（split） 。Master节点会尽量将任务分配到离输入分片较近的节点上执行， 以减少网络通信量。',
  datetime('now'), 1, 1, 0, 0, 2, 2, datetime('now')
);
-----------------------------------------------------------------------------------------------------------------------
INSERT OR REPLACE INTO main.qa_question (id, description, pub_date, reward, closed, user_id, negative_votes, positive_votes, total_points, slug, title)
VALUES (
  2,
  'mysql数据量太大，算一个存储过程特别慢，是否可以用spark替代存储过程，但是spark写数据库只有append 或overwrite。没有update。 并且spark可以保证大事务，要么都写成功要么失败。',
  datetime('now'), 0, 0, 2, 0, 0, 0,
  'Spark可否替代存储过程', 'Spark可否替代存储过程'
);
INSERT OR REPLACE INTO main.taggit_taggeditem (object_id, content_type_id, tag_id) VALUES (
  2, 13, 1
);
INSERT OR REPLACE INTO main.qa_answer (answer_text, pub_date, question_id, user_id, answer, negative_votes, positive_votes, total_points, updated)
VALUES (
  'Spark不支持细粒度更新，无法满足这种场景要求',
  datetime('now'), 2, 1, 0, 0, 1, 1, datetime('now')
);
-----------------------------------------------------------------------------------------------------------------------
INSERT OR REPLACE INTO main.qa_question (id, description, pub_date, reward, closed, user_id, negative_votes, positive_votes, total_points, slug, title)
VALUES (
  3,
  '如果是的话，只需要再次执行hdfs namenode -format就可以了吗？',
  datetime('now'), 0, 0, 2, 0, 0, 0,
  '配置好集群以后如果要再增加节点是否要重新格式化namenode', '配置好集群以后如果要再增加节点是否要重新格式化namenode？'
);
INSERT OR REPLACE INTO main.taggit_taggeditem (object_id, content_type_id, tag_id) VALUES (
  3, 13, 1
);
INSERT OR REPLACE INTO main.qa_answer (id, answer_text, pub_date, question_id, user_id, answer, negative_votes, positive_votes, total_points, updated)
VALUES (
  3, '在NameNode上刷新节点即可',
  datetime('now'), 3, 1, 0, 0, 1, 1, datetime('now')
);
-- answer comment
INSERT OR REPLACE INTO main.qa_answercomment (comment_text, answer_id, user_id, pub_date) VALUES (
  '直接执行hdfs dfsadmin -refreshNodes就可以了吗', 3, 2, datetime('now')
);
-----------------------------------------------------------------------------------------------------------------------
INSERT OR REPLACE INTO main.qa_question (id, description, pub_date, reward, closed, user_id, negative_votes, positive_votes, total_points, slug, title)
VALUES (
  4,
  '书上写的是SSH这个SSL是什么意思',
  datetime('now'), 0, 0, 2, 0, 0, 0,
  'SSL是什么', 'SSL是什么'
);
INSERT OR REPLACE INTO main.taggit_taggeditem (object_id, content_type_id, tag_id) VALUES (
  4, 13, 1
);
INSERT OR REPLACE INTO main.qa_answer (answer_text, pub_date, question_id, user_id, answer, negative_votes, positive_votes, total_points, updated)
VALUES (
  'SSH 为 Secure Shell 的缩写，可以理解为远程安全命令行。

SSL(Secure Sockets Layer 安全套接层),及其继任者传输层安全（Transport Layer Security，TLS）是为网络通信提供安全及数据完整性的一种安全协议。TLS与SSL在传输层对网络连接进行加密。',
  datetime('now'), 4, 1, 0, 0, 2, 2, datetime('now')
);
-----------------------------------------------------------------------------------------------------------------------
INSERT OR REPLACE INTO main.qa_question (id, description, pub_date, reward, closed, user_id, negative_votes, positive_votes, total_points, slug, title)
VALUES (
  5,
  'mapReduce适合处理那些可以进行分布式运算的任务。那么，如果有一个任务不能进行分布式运算，mapReduce将如何处理呢？能举个具体的例子说明一下吗？',
  datetime('now'), 0, 0, 2, 0, 0, 0,
  '有没有mapReduce无法处理的任务', '有没有mapReduce无法处理的任务？'
);
INSERT OR REPLACE INTO main.taggit_taggeditem (object_id, content_type_id, tag_id) VALUES (
  5, 13, 1
);
INSERT OR REPLACE INTO main.qa_answer (answer_text, pub_date, question_id, user_id, answer, negative_votes, positive_votes, total_points, updated)
VALUES (
  'mapreduce模型表达能力并不是无限的，有一些任务是无法用mapreduce处理的（其中一部分用spark可以处理）。如果不能分布式运算就只能单机运算咯。',
  DATETIME('now'), 5, 1, 0, 0, 2, 2, datetime('now')
);
-----------------------------------------------------------------------------------------------------------------------
INSERT OR REPLACE INTO main.qa_question (id, description, pub_date, reward, closed, user_id, negative_votes, positive_votes, total_points, slug, title)
VALUES (
  6,
  '如果namenode对应的是master机器，datanode对应的是slave机器。',
  datetime('now'), 0, 0, 2, 0, 0, 0,
  'secondary_namenode对应的是哪一台机器呢', 'secondary namenode对应的是哪一台机器呢？'
);
INSERT OR REPLACE INTO main.taggit_taggeditem (object_id, content_type_id, tag_id) VALUES (
  6, 13, 1
);
INSERT OR REPLACE INTO main.qa_answer (answer_text, pub_date, question_id, user_id, answer, negative_votes, positive_votes, total_points, updated)
VALUES (
  'secondary namenode是一种备份机制，应该是在一台新机器上。',
  DATETIME('now'), 6, 1, 0, 0, 1, 1, datetime('now')
);
-----------------------------------------------------------------------------------------------------------------------
INSERT OR REPLACE INTO main.qa_question (id, description, pub_date, reward, closed, user_id, negative_votes, positive_votes, total_points, slug, title)
VALUES (
  7,
  '',
  datetime('now'), 0, 0, 2, 0, 0, 0,
  '在运行Hadoop第一个例子的时候出现了权限不够', '在运行Hadoop第一个例子的时候出现了权限不够'
);
INSERT OR REPLACE INTO main.taggit_taggeditem (object_id, content_type_id, tag_id) VALUES (
  7, 13, 1
);
INSERT OR REPLACE INTO main.qa_answer (answer_text, pub_date, question_id, user_id, answer, negative_votes, positive_votes, total_points, updated)
VALUES (
  '你有给hadoop用户添加权限吗？如果添加了还是不行，你在命令前加一个sudo试试',
  DATETIME('now'), 7, 1, 0, 0, 1, 1, datetime('now')
);
-----------------------------------------------------------------------------------------------------------------------
INSERT OR REPLACE INTO main.qa_question (id, description, pub_date, reward, closed, user_id, negative_votes, positive_votes, total_points, slug, title)
VALUES (
  8,
  '我有一个问题一直不能很明白：就是Hive(数据仓库)和Sql什么关系呢？感觉Hive就是Sql，只不过也是架构在hadoop上，然后把Hql语句转换为MapReduce来实现的，其数据存储的结构也是sql结构吧？所以想求问一下Hive和Sql什么关系和区别呢？',
  datetime('now'), 0, 0, 2, 0, 0, 0,
  'Hive和sql什么关系呢', 'Hive和sql什么关系呢？'
);
INSERT OR REPLACE INTO main.taggit_taggeditem (object_id, content_type_id, tag_id) VALUES (
  8, 13, 1
);
INSERT OR REPLACE INTO main.qa_answer (answer_text, pub_date, question_id, user_id, answer, negative_votes, positive_votes, total_points, updated)
VALUES (
  'Hive中不能进行删除和修改数据，sql不能对海量数据进行MapReduce处理，Hive和sql连用可以在Hive中对海量数据进行处理后用Sqoop导入其它数据库，然后进行增删改查的处理',
  DATETIME('now'), 8, 1, 0, 0, 3, 3, datetime('now')
);