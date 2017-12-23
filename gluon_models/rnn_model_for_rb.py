import logging
import math
import random
import time

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import nn, rnn

from csv_data.read_csv_data import get_simple_data, get_data_ma_smooth

MODEL_NAME = __name__
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    filename=MODEL_NAME + '.log',
                    filemode='w')
logger = logging.getLogger(__name__)

BATCH_SIZE = 128
SEQ_LEN = 500
PREDICT_LEN = 1
NUM_LAYERS = 4
HIDDEN_UNITS = 128
NUM_CLASSES = 2
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
ATTN_LENGTH = 20
DROPOUT = 0.1
EPOCH = 20
EXAMPLES = 280001
ITER_NUM_EPCOH = int(EXAMPLES / BATCH_SIZE)
DECAY_STEP = ITER_NUM_EPCOH * EPOCH
LR = 0.04
END_LR = 0.0002
INPUT_SIZE = 5
CELL_TYPE = 'rnn_tanh'
CSV_FILE = '/home/daiab/machine_disk/code/quantum/csv_data/RB_min.csv'
RESTORE_PATH = './model_save/%s.params' % MODEL_NAME
# infer
INFER_SIZE = 10


class RNNClsModel(gluon.Block):
    def __init__(self, mode,
                 num_embed,
                 num_hidden,
                 seq_len, num_layers,
                 dropout=0.0,
                 **kwargs):
        super(RNNClsModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            # self.emb = nn.Embedding(vocab_size, num_embed,
            #                         weight_initializer=mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, activation='relu', num_layers=num_layers,
                                   layout='NTC', dropout=dropout, input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers=num_layers, layout='NTC', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers=num_layers, layout='NTC', dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers=num_layers, layout='NTC', dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru" % mode)

            self.fc = nn.Dense(NUM_CLASSES, in_units=num_hidden * seq_len)
            self.num_hidden = num_hidden
            self.seq_len = seq_len

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        output = self.drop(output)
        decoded = self.fc(output.reshape((-1, self.num_hidden * self.seq_len)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class LRDecay:
    def __init__(self, base_lr, end_lr, power, total_step):
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.cur_step = 0
        self.total_step = total_step
        self.power = power

    @property
    def lr(self):
        self.cur_step += 1
        lr = (1 - self.cur_step / self.total_step) ** self.power * (self.base_lr - self.end_lr) + self.end_lr
        if lr < self.end_lr:
            return self.end_lr
        return lr


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


class DataIter:
    def __init__(self, csv_file, seq_len, batch_size, predict_len):
        self._X, self._Y = get_simple_data(csv_file, seq_len, predict_len)
        self._X = [d.values for d in self._X]
        size = len(self._X)
        self._count = 0
        self._batch_size = batch_size
        self._idx = [(start, start + batch_size) for start in range(size - batch_size)]
        self._size = len(self._idx)
        logger.info('all batch size == %s', self._size)
        random.shuffle(self._idx)

    def next(self):
        start, end = random.choice(self._idx)
        data_batch = self._X[start: end]
        label_batch = self._Y[start: end]
        return data_batch, label_batch


def train():
    data_iter = DataIter(CSV_FILE, SEQ_LEN, BATCH_SIZE, predict_len=40)
    print('read data over')
    LOG_INTERVAL = 20
    CLIP = 0.2
    context = mx.gpu(0)
    model = RNNClsModel(mode=CELL_TYPE, num_embed=INPUT_SIZE,
                        num_hidden=HIDDEN_UNITS, seq_len=SEQ_LEN,
                        num_layers=NUM_LAYERS, dropout=DROPOUT)
    model.collect_params().initialize(mx.init.Xavier(), ctx=context)
    # model.collect_params().load(RESTORE_PATH, context)
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': LR,
                             'momentum': 0.9,
                             'wd': 0.0})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_decay = LRDecay(LR, END_LR, 0.6, DECAY_STEP)
    for epoch in range(EPOCH):
        total_loss = 0.0
        total_acc = 0.0
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=context)
        for i in range(ITER_NUM_EPCOH):
            data, target = data_iter.next()
            data = mx.nd.array(data, context)
            target = mx.nd.array(target, context)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, CLIP * BATCH_SIZE)

            trainer.step(BATCH_SIZE)
            total_loss += mx.nd.sum(L).asscalar()
            total_acc += mx.nd.sum(mx.nd.equal(mx.nd.argmax(output, axis=1), target)).asscalar()

            if i % LOG_INTERVAL == 0:
                cur_loss = total_loss / BATCH_SIZE / LOG_INTERVAL
                cur_acc = total_acc / BATCH_SIZE / LOG_INTERVAL
                logger.info('%d # %d loss %.5f, ppl %.5f, lr %.5f, acc %.5f',
                            epoch, i, cur_loss, math.exp(cur_loss), trainer._optimizer.lr, cur_acc)
                total_loss = 0.0
                total_acc = 0.0
            trainer._optimizer.lr = lr_decay.lr
        model.collect_params().save(RESTORE_PATH)


if __name__ == '__main__':
    train()
