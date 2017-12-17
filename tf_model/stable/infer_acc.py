import logging

import numpy as np
import tensorflow as tf
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
        # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        # print(self._sess.run('fully_connected_1/biases:0'))
        # exit()

    def step(self, data):
        output = self._sess.run(self._graph, feed_dict={self._placeholder: data})
        return output


def init(context):
    # 在context中保存全局变量
    context.all_code = all_instruments('CS').order_book_id.values
    context.infer_mode = Infer(RESTORE_PATH)
    context.right = 0
    context.total_count = 0


def before_trading(context):
    fields = ["open", "close", "high", "low", "total_turnover", "volume"]
    context.bi = []
    for code in context.all_code:
        # prices = history_bars(context.s1, context.LONGPERIOD + 1, '1d', 'close')
        data = history_bars(code, bar_count=SEQ_LEN + 1, frequency='1d', fields=fields)
        if data.shape[0] < SEQ_LEN + 1:
            continue
        data = _norm_max_min(np.array(data.tolist(), subok=True))
        data = np.array([data])
        softmax_output = context.infer_mode.step(data[:, :-1, :])
        up = np.argmax(softmax_output, axis=1)[-1]
        if softmax_output[-1][0] > 0.95 or softmax_output[-1][0] < 0.05:
            context.total_count += 1
            if up == int(data[0, -1, 1] / data[0, -2, 1] > 1):
                context.right += 1
    if context.total_count == 0:
        print('total count == 0')
    else:
        print('base == %s, right ratio == %s' % (context.total_count, context.right / context.total_count))
    context.total_count = 0
    context.right = 0


def handle_bar(context, bar_dict):
    pass


def after_trading(context):
    print('========== over ==========')

# rqalpha run -s 2017-09-12 -e 2017-10-05 -f /home/daiab/machine_disk/code/quantum/stable/infer_acc.py  --account stock 100000 -bm 000001.XSHE


