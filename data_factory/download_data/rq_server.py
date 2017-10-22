from socket import *
import pickle
import logging
import pandas as pd
import struct
import os
from multiprocessing import Queue
from threading import Thread
import time
import datetime

logging.basicConfig(filename='run.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

FREQ_1D_FIELDS = ['total_turnover', 'low', 'close', 'open', 'high', 'volume']


DATA_PATH = '/root/share_data'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

data_queue = Queue()


def save_data(data_q):
    k = None
    while True:
        try:
            if data_q.qsize() == 0:
                time.sleep(0.02)
                continue
            data = data_q.get()
            logging.info('recv data length == %s', len(data))
            data = pickle.loads(data)
            if data['type'] == 'get_price':
                pd_data = data['value']
                code = data['code']
                freq = data['freq']
                if freq == '1d':
                    time_list = pd_data.index.values
                    pd_data = pd_data.ix[:, FREQ_1D_FIELDS]
                    list_data = pd_data.values.tolist()
                    for i, t in enumerate(time_list):
                        t = pd.Timestamp(t).to_pydatetime()
                        d = list_data[i]
                        d.append(code).append(t)
                        # TODO: insert into db
                        logging.info(d)
                elif freq == '30m':
                    pass
                else:
                    raise Exception('freq error %s' % freq)
            else:
                for k in data.keys():
                    file_path = os.path.join(DATA_PATH, '%s.pkl' % k)
                    pickle.dump(data, open(file_path, 'wb'))
        except:
            logging.exception('error %s' % k)


th = Thread(target=save_data, args=(data_queue,))
th.daemon = True
th.start()

HOST = '0.0.0.0'
PORT = 5000
BUFSIZ = 1024
ADDR = (HOST, PORT)
sock = socket(AF_INET, SOCK_STREAM)
sock.bind(ADDR)
sock.listen(5)
while True:
    print('waiting for connection')
    tcpClientSock, addr = sock.accept()
    print('connect from ', addr)
    try:
        data_len = tcpClientSock.recv(4)
        data_len = struct.unpack('>I', data_len)[0]
        print(data_len)
        data = b''
        while len(data) < data_len:
            data += tcpClientSock.recv(data_len)
        data_queue.put(data)
        tcpClientSock.send(b'ok')
    except Exception as e:
        logging.exception(e)
    finally:
        tcpClientSock.close()
sock.close()
