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


def get_ohlcv(start_date, end_date, code, fields=FREQ_1D_FIELDS):
    sql = "select " + "{} " * len(fields) + " from get_price where trade_date >= '%s' and trade_date <= '%s' \
            and code = '%s' order by trade_date asc"
    sql = sql.format(*fields)
    sql = sql % (start_date, end_date, code)
    return conn_manager.query_sql(sql)


def get_previous_returns(start_date, end_date, code, pre_days='2d'):
    pass