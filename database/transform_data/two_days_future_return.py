import pymysql
import time
import numpy as np
from database import get_all_code

DAYS = 2
DATABASE = 'rqalpha'


class ConnManage:
    def __init__(self, database):
        self._time_interval = 60 * 5  # 5 min
        self._start_time = time.time()
        self._host = 'localhost'
        self._user = 'root'
        self._passwd = ''
        self._database = database
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

    def _create_conn(self):
        return pymysql.connect(host=self._host, user=self._user, password=self._passwd, database=self._database)


conn_manager = ConnManage(DATABASE)


def cal_return(code, days):
    sql = "select close, trade_date from get_price where code = '%s' order by trade_date asc" % code
    query_result = conn_manager.query_sql(sql)
    if len(query_result) == 0:
        print('code: %s select result is null' % code)
        return
    close_p = []
    trade_date_list = []
    for items in query_result:
        close_p.append(items[0])
        trade_date_list.append(items[1])
    close_p = np.array(close_p, dtype=np.float32)
    trade_date_list = trade_date_list[:-days]
    returns = close_p[days:] / close_p[:-days]
    for i, trade_date in enumerate(trade_date_list):
        ret = returns[i]
        sql = "update get_price set future_two_day_returns={:.10f} " \
              " where code = '{}' and trade_date = '{}'".format(
                ret, code, trade_date)
        conn_manager.exec_sql(sql)


if __name__ == '__main__':
    all_code = get_all_code(conn_manager)
    count = 0
    for code in all_code:
        print(code)
        print(count)
        cal_return(code, DAYS)
        count += 1
