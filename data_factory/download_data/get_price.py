# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
import pickle
import time
import struct
from socket import *
import pandas

SEND_SERVER = '47.92.110.187'
PORT = 5000
FREQ = '1d'
START_DATE = '2006-01-04'
END_DATE = '2017-09-09'
TYPE = 'get_price'


def send_data(data):
    data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    data_len = struct.pack('>I', len(data))
    print(len(data))
    data = data_len + data
    client = socket(AF_INET, SOCK_STREAM)
    client.connect((SEND_SERVER, PORT))
    client.sendall(data)
    client.close()


def get_data():
    now = time.time()
    all_code = all_instruments('CS').order_book_id.values
    for i, code in enumerate(all_code):
        try:
            share_data = get_price(code, start_date=START_DATE, end_date=END_DATE,
                                   frequency=FREQ, fields=None,
                                   adjust_type='pre', skip_suspended=True)
            share_data = pandas.DataFrame(share_data)
            send_data({'value': share_data, 'code': code, 'freq': FREQ, 'type': TYPE})
            logger.info('# %s, code: %s time: %s' % (i, code, time.time() - now))
        except:
            logger.exception('%s error' % code)


def init(context):
    logger.info(pandas.__version__)
    context.s1 = '000001.XSHE'
    get_data()


def before_trading(context):
    pass


def handle_bar(context, bar_dict):
    order_shares(context.s1, 1000)


def after_trading(context):
    pass
