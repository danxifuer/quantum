from socket import *
import pickle
import logging
import struct
import os

logging.basicConfig(filename='run.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


DATA_PATH = '/root/share_data'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)


def save_data(data):
    k = None
    try:
        print(len(data))
        data = pickle.loads(data)
        for k in data.keys():
            file_path = os.path.join(DATA_PATH, '%s.pkl' % k)
            pickle.dump(data, open(file_path, 'wb'))
    except:
        logging.exception('error %s' % k)


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
        save_data(data)
        tcpClientSock.send(b'ok')
    except Exception as e:
        logging.exception(e)
    finally:
        tcpClientSock.close()
sock.close()
