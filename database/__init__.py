import pymysql
import time


def get_all_code(conn_manager):
    sql = 'select code from base_info'
    code = conn_manager.query_sql(sql)
    return [c[0] for c in code]

DATABASE = 'quantum'


class ConnManage:
    def __init__(self):
        self._time_interval = 60 * 5  # 5 min
        self._start_time = time.time()
        self._host = '116.196.115.222'
        self._user = 'daiab'
        self._passwd = 'asdf..12'
        self._database = DATABASE
        self._conn = self._create_conn()

    def _get_conn(self):
        if time.time() - self._start_time > self._time_interval:
            self._conn.close()
            self._conn = self._create_conn()
            self._start_time = time.time()
        return self._conn

    def exec_sql(self, sql):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()

    def query_sql(self, sql):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    def query_template(self, table, fileds, query_dict):
        q_field = ', '.join(fileds)
        cond = []
        for k, v in query_dict.items():
            cond.append('%s = %s' % (k, v))
        cond = ' and '.join(cond)
        sql = 'select %s from %s where %s' % (q_field, table, cond)
        return sql

    def exec_template(self, table, insert_dict):
        keys = ['%s' % k for k in insert_dict.keys()]
        values = ['%s' % v for v in insert_dict.values()]
        sql = 'insert into %s (%s) values (%s)' % (table, ', '.join(keys), ', '.join(values))
        return sql

    def _create_conn(self):
        return pymysql.connect(host=self._host, user=self._user, password=self._passwd, database=self._database)


def exec_template(table, insert_dict):
    keys = ['%s' % k for k in insert_dict.keys()]
    values = ['%s' % v for v in insert_dict.values()]
    sql = 'insert into %s (%s) values (%s)' % (table, ', '.join(keys), ', '.join(values))
    return sql

if __name__ == '__main__':
    print(exec_template('daiab', {'daaib': '"asdf"',  'dasdf': 123}))