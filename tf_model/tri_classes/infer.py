import logging

import numpy as np
import tensorflow as tf
from rqalpha.api import order_shares
from rqalpha.api.api_base import all_instruments, history_bars

from data_process import _norm_max_min
from model import get_model
from rnn_config import *

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
                    # filename='parser_result.log',
                    # filemode='w')


class Infer:
    def __init__(self, restore_path):
        self._placeholder = tf.placeholder(dtype=tf.float32, shape=(1, SEQ_LEN, INPUT_SIZE))
        self._graph = get_model(self._placeholder, None, False)
        self._sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self._sess, restore_path)

    def step(self, data):
        output = self._sess.run(self._graph, feed_dict={self._placeholder: data})
        return output


def cal_acc():
    pass


def init(context):
    # 在context中保存全局变量
    context.all_code = all_instruments('CS').order_book_id.values
    context.infer_mode = Infer(RESTORE_PATH)


def before_trading(context):
    fields = ["open", "close", "high", "low", "total_turnover", "volume"]
    context.bi = []
    for code in context.all_code:
        # prices = history_bars(context.s1, context.LONGPERIOD + 1, '1d', 'close')
        data = history_bars(code, bar_count=SEQ_LEN, frequency='1d', fields=fields)
        data = _norm_max_min(np.array(data.tolist(), subok=True))
        data = np.array([data])
        if len(data.shape) < 3 or data.shape[1] < SEQ_LEN:
            continue
        softmax_output = context.infer_mode.step(data)
        up = np.argmax(softmax_output, axis=1)[-1]
        if up and softmax_output[-1][1] > 0.9:
            context.bi.append(code)
    print('bi size == %d' % len(context.bi))


def handle_bar(context, bar_dict):
    num = len(context.bi)
    print('filter num = %d' % num)
    for code in context.all_code:
        position = context.portfolio.positions.get(code, None)
        if position and position.quantity > 0:
            order_shares(code, - position.quantity)
        if code in context.bi:
            shares = context.portfolio.cash / bar_dict[code].close / num
            print(int(shares))
            order_shares(code, int(shares))


def after_trading(context):
    print('========== over ==========')


# rqalpha run -s 2016-01-01 -e 2016-02-01 -f /home/daiab/machine_disk/code/Craft/rnn/atom72/infer.py  --account stock 100000 -p -bm 000001.XSHE
