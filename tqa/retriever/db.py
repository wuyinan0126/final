import unicodedata

import pymysql
import re

from tqa import DEFAULTS
from tqa.retriever.utils import normalize


class Db(object):
    """ Sqlite数据库对象 """

    def __init__(self, **kwargs):
        self.conn = pymysql.connect(
            DEFAULTS['db_host'], DEFAULTS['db_user'], DEFAULTS['db_password'], DEFAULTS['db_database'],
            use_unicode=True, charset="utf8"
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the connection to the database."""
        self.conn.close()

    def get_ids(self, db_table):
        """ 获取数据库中所有id """
        cursor = self.conn.cursor()
        cursor.execute("SELECT document_id FROM %s" % (db_table,))
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_text(self, db_table, document_id):
        """ 根据document_id获得对应text """
        cursor = self.conn.cursor()
        # 注意最后的','传递tuple
        cursor.execute(
            "SELECT document_text FROM %s WHERE document_id = '%s'" %
            (db_table, normalize(document_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_tokens(self, db_table, document_id):
        """ 根据document_id获得对应tokens """
        cursor = self.conn.cursor()
        # 注意最后的','传递tuple
        # print("SELECT * FROM %s WHERE document_id = '%s'" % (db_table, normalize(document_id),))
        document_id = re.sub("'", "\\'", document_id)
        cursor.execute(
            "SELECT * FROM %s WHERE document_id = '%s'" %
            (db_table, normalize(document_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result
