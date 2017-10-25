import pymysql
import time

FREQ_1D_FIELDS = ['open', 'high', 'low', 'close', 'volume']
HOST = '0.0.0.0'
PORT = 5000
BUFSIZ = 1024
ADDR = (HOST, PORT)
DATABASE = 'rqalpha'


class ConnManage:
    def __init__(self, database):
        self._time_interval = 60 * 5  # 5 min
        self._start_time = time.time()
        self._host = 'localhost'
        self._user = 'root'
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
            and code = '%s' order by trade_date asc"
    sql = sql % (start_date, end_date, code)
    return conn_manager.query_sql(sql)


def get_ohlcv(code):
    sql = "select open, high, low, close, volume from get_price where \
            and code = '%s' order by trade_date asc" % code
    return conn_manager.query_sql(sql)


def get_ohlcv_previous_returns(code, pre_days='2d'):
    if pre_days == '2d':
        sql = 'select p.open, p.high, p.low, p.close, p.volume, r.close_return from get_price as p ' \
              ' join pre_two_day_returns as r on where r.trade_date = p.trade_date and ' \
              ' code = "%s" order by p.trade_date asc' % code
    else:
        raise NotImplementedError
    return conn_manager.query_sql(sql)


def get_future_returns(start_date, end_date, code, future_days='1d'):
    if future_days == '1d':
        sql = 'select close_return from future_one_day_returns where' \
              ' trade_date >= "%s" and trade_date <= "%s" and code = "%s"' % (start_date, end_date, code)
    else:
        raise NotImplementedError
    return conn_manager.query_sql(sql)


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
    return conn_manager.query_sql(sql)


