import pickle
import time
import struct
from socket import *
import pandas


def send_data(data):
    data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    data_len = struct.pack('>I', len(data))
    print(len(data))
    data = data_len + data
    client = socket(AF_INET, SOCK_STREAM)
    client.connect(('47.92.110.187', 5000))
    client.sendall(data)
    client.close()


def get_data():
    now = time.time()
    all_code = all_instruments('CS').order_book_id.values
    all_code = all_code.split(',')
    # print(all_code.shape)
    # print(','.join(all_code.tolist()))
    for i, code in enumerate(all_code):
        try:
            share_data = get_price(code, start_date='2006-01-04', end_date='2017-09-09', frequency='30m', fields=None,
                                   adjust_type='pre', skip_suspended=True)
            share_data = pandas.DataFrame(share_data)
            send_data({code: share_data})
            logger.info('# %s, code: %s time: %s' % (i, code, time.time() - now))
        except:
            logger.exception('%s error' % code)


def init(context):
    get_data()
    logger.info(pandas.__version__)
    context.s1 = "000001.XSHE"
    logger.info("RunInfo: {}".format(context.run_info))


def before_trading(context):
    pass


def handle_bar(context, bar_dict):
    order_shares(context.s1, 1000)


def after_trading(context):
    pass
