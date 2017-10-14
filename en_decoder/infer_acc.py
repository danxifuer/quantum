import tensorflow as tf
import numpy as np
from rnn_config import *
from model import get_model
from data_process import _norm_max_min
from rqalpha.api.api_base import all_instruments, history_bars
# from rqalpha.api import order_shares
from tensorflow.python.framework import ops
from tensorflow.contrib.seq2seq.python.ops.helper import Helper
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')


# filename='parser_result.log',
# filemode='w')

class GreedyEmbeddingHelper(Helper):
    def __init__(self, embedding, start_tokens, end_token):
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = tf.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embedding_fn(self._start_tokens)

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        sample_ids = tf.cast(
            tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self._end_token)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)


class Infer:
    def __init__(self, restore_path):
        self._placeholder = tf.placeholder(dtype=tf.float32, shape=(1, SEQ_LEN, INPUT_SIZE))
        self._graph = get_model(self._placeholder, None, False)
        self._sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self._sess, restore_path)

    def step(self, data):
        output = self._sess.run(self._graph, feed_dict={self._placeholder: data})
        print(output.shape)
        return output


def init(context):
    # 在context中保存全局变量
    context.all_code = all_instruments('CS').order_book_id.values
    context.infer_mode = Infer(RESTORE_PATH)


def before_trading(context):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~%s~~~~~~~~~~~~~~~~~~~~~~~~~~~' % context.now)
    fields = ["open", "close", "high", "low", "total_turnover", "volume"]
    context.bi = []
    filter_prob = []
    for code in context.all_code:
        data = history_bars(code, bar_count=SEQ_LEN + 1, frequency='1d', fields=fields)
        if data.shape[0] < SEQ_LEN + 1:
            continue
        data = np.array(data.tolist(), subok=True)
        # print(data[-1, 1])
        norm_data = _norm_max_min(data, copy=True)
        norm_data = np.array([norm_data])
        softmax_output = context.infer_mode.step(norm_data[:, :-1, :])
        up_ratio = data[-1, 1] / data[-2, 1]
        real = int(up_ratio > 1)
        if softmax_output[-1][1] > 0.95:  # up
            filter_prob.append((softmax_output[-1][1], 1, real, (up_ratio, data[-1, 1], code.split('.')[0])))
        elif softmax_output[-1][0] > 0.95:  # down
            filter_prob.append((softmax_output[-1][0], 0, real, (up_ratio, data[-1, 1], code.split('.')[0])))
    filter_prob = sorted(filter_prob, key=lambda x: -x[0])
    if len(filter_prob) > 10:
        filter_prob = filter_prob[:10]
    up_right = 0
    up_total = 0
    down_right = 0
    down_total = 0
    for item in filter_prob:
        if item[1] == 1:
            if item[1] == item[2]:
                up_right += 1
            up_total += 1
        else:
            if item[1] == item[2]:
                down_right += 1
            down_total += 1
    if down_total == 0 or up_total == 0:
        print('error and passed, down_total: %s, up_total: %s' % (down_total, up_total))
        return
    print('## up == (%s, %s) down == (%s, %s)' % (up_total,
                                                  up_right / up_total,
                                                  down_total,
                                                  down_right / down_total))
    for i in filter_prob:
        print(i)


def handle_bar(context, bar_dict):
    pass


def after_trading(context):
    pass

# export PYTHONPATH=/home/daiab/machine_disk/code/quantum/atom72:$PATHONPATH
# rqalpha run -s 2017-09-12 -e 2017-10-5 -f
# /home/daiab/machine_disk/code/quantum/atom72/infer_acc.py
# --account stock 100000 -bm 000001.XSHE
