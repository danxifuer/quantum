import pymysql
import time
import logging
import numpy as np
head = '%(asctime)-15s %(message)s'
console = logging.StreamHandler()
logging.basicConfig(level=logging.DEBUG, format=head, handlers=[console])


HOST = '0.0.0.0'
PORT = 5000
BUFSIZ = 1024
ADDR = (HOST, PORT)
DATABASE = 'rqalpha'


class ConnManage:
    def __init__(self, database):
        self._time_interval = 60 * 5  # 5 min
        self._start_time = time.time()
        self._host = '116.196.115.222'
        self._user = 'daiab'
        self._passwd = 'asdf..12'
        self._database = database
        self._conn = self._create_conn()

    def _get_conn(self):
        if time.time() - self._start_time > self._time_interval:
            self._conn.close()
            self._conn = self._create_conn()
            self._start_time = time.time()
        return self._conn

    def query_sql(self, sql):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    def _create_conn(self):
        return pymysql.connect(host=self._host, user=self._user, password=self._passwd, database=self._database)


conn_manager = ConnManage(DATABASE)


def get_ohlcv_by_date(start_date, end_date, code):
    sql = "select open, high, low, close, volume from get_price where trade_date >= '%s' and trade_date <= '%s' \
            and code = '%s' order by id asc"
    sql = sql % (start_date, end_date, code)
    ret = conn_manager.query_sql(sql)
    if len(ret) == 0:
        return None
    return np.array(ret, dtype=np.float32)


def get_ohlcv(code):
    sql = "select open, high, low, close, volume from get_price where \
            code = '%s' order by id asc" % code
    ret = conn_manager.query_sql(sql)
    if len(ret) == 0:
        return None
    return np.array(ret, dtype=np.float32)


def get_ohlcv_pre_ret(code, pre_days='2d'):
    if pre_days == '2d':
        sql = 'select open, high, low, close, volume, pre_two_day_returns ' \
              ' from get_price ' \
              ' where code = "%s" order by id asc' % code
    else:
        raise NotImplementedError
    ret = conn_manager.query_sql(sql)
    if len(ret) == 0:
        return None
    return np.array(ret, dtype=np.float32)


def get_ohlcv_future_ret(code, from_date, end_date, future_days='1d'):
    logging.info('get_ohlcv_future_ret, code: %s', code)
    assert future_days == '1d', 'NotImplementError'
    sql = 'select open, high, low, close, volume, future_one_day_returns ' \
          ' from get_price ' \
          ' where code = "%s" and trade_date >= "%s" and trade_date <= "%s" ' \
          ' order by id asc' % (code, from_date, end_date)
    ret = conn_manager.query_sql(sql)
    if len(ret) == 0:
        return None
    return np.array(ret, dtype=np.float32)


def get_normed_ohlcv_future_ret(code, from_date, end_date, future_days='1d'):
    logging.info('get_ohlcv_future_ret, code: %s', code)
    assert future_days == '1d', 'NotImplementError'
    sql = 'select h_o, l_o, c_o, o_c, h_c, l_c, volume, future_one_day_returns ' \
          ' from norm_data_across_stock ' \
          ' where code = "%s" and trade_date >= "%s" and trade_date <= "%s" ' \
          ' order by id asc' % (code, from_date, end_date)
    ret = conn_manager.query_sql(sql)
    if len(ret) == 0:
        return None
    return np.array(ret, dtype=np.float32)


def get_code(from_date=None, end_date=None, greater_days=200):
    '''
    :param from_date:
    :param end_date:
    :param greater_days:  must passed
    :return:
    '''
    sql = 'select code from base_info where total_num >= %s ' % greater_days
    if from_date:
        sql += ' and from_date >= "%s" ' % from_date
    if end_date:
        sql += ' and end_date <= "%s" ' % from_date
    cl = conn_manager.query_sql(sql)
    return [c[0] for c in cl]


if __name__ == '__main__':
    code_list = get_code()
    print(code_list[0])
    data = get_ohlcv_future_ret(code_list[0])
    print(data[0:2])
