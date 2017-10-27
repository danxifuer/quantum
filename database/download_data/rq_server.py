from socket import *
import pickle
import logging
import pandas as pd
import struct
import pymysql
from multiprocessing import Queue
from threading import Thread
import time
import datetime

#  pip3 install PyMySQL

logging.basicConfig(filename='run.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

FREQ_1D_FIELDS = ['open', 'high', 'low', 'close', 'total_turnover', 'volume']
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

    def exec_sql(self, sql):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()

    def _create_conn(self):
        return pymysql.connect(host=self._host, user=self._user, password=self._passwd, database=self._database)


class Server:
    def __init__(self):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.bind(ADDR)
        sock.listen(5)
        self.sock = sock
        self.data_q = Queue()
        self.conn_manager = ConnManage(DATABASE)
        th = Thread(target=self.handle_data)
        th.daemon = True
        th.start()
        self.run()

    def run(self):
        while True:
            print('waiting for connection')
            client_sock, addr = self.sock.accept()
            try:
                data_len = client_sock.recv(4)
                data_len = struct.unpack('>I', data_len)[0]
                print('recv data len: %s', data_len)
                data = b''
                while len(data) < data_len:
                    data += client_sock.recv(data_len)
                self.data_q.put(data)
                client_sock.send(b'ok')
            except Exception as e:
                logging.exception(e)
            finally:
                client_sock.close()

    def handle_data(self):
        k = None
        count = 0
        while True:
            try:
                if self.data_q.qsize() == 0:
                    time.sleep(0.02)
                    continue
                count += 1
                if count % 100 == 0:
                    print('#%s th code ready to write' % count)
                data = self.data_q.get()
                data = pickle.loads(data)
                if data['type'] == 'get_price':
                    pd_data = data['value']
                    code = data['code']
                    freq = data['freq']
                    print('recv stock code == %s', code)
                    if freq == '1d':
                        time_list = pd_data.index.values
                        pd_data = pd_data.ix[:, FREQ_1D_FIELDS]
                        list_data = pd_data.values.tolist()
                        for i, t in enumerate(time_list):
                            t = pd.Timestamp(t).to_pydatetime()
                            d = list_data[i]
                            d.append(code)
                            d.append(t)
                            # TODO: insert into db
                            sql = '''insert into get_price \
                                    (open, high, low, close, total_turnover, volume, code, trade_date) \
                                    values (%s, %s, %s, %s, %s, %s, '%s', '%s') \
                                  ''' % tuple(d)
                            self.conn_manager.exec_sql(sql)
                    elif freq == '30m':
                        pass
                    else:
                        raise Exception('freq error %s' % freq)
                else:
                    logging.error('data type error %s', data['type'])
            except:
                logging.exception('error %s' % k)


if __name__ == '__main__':
    s = Server()
