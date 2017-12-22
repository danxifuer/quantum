from mxnet import gluon
from mxnet.gluon import nn, rnn
from mxnet import autograd
from get_db_data.tools import concat_day_min
import mxnet as mx
import math
import time
import pandas as pd
import numpy as np
import logging
import random

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
EXAMPLES = 58555
ITER_NUM_EPCOH = int(EXAMPLES / BATCH_SIZE)
DECAY_STEP = ITER_NUM_EPCOH * EPOCH
LR = 0.04
END_LR = 0.0002
INPUT_SIZE = 5
CELL_TYPE = 'rnn_tanh'
CSV_FILE = '/home/daiab/machine_disk/code/quantum/database/RB_5min.csv'
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


def read_csv(csv_file):
    rb = pd.read_csv(csv_file, index_col=0)
    rb.index = pd.DatetimeIndex(rb.index)
    close = (rb.loc[:, 'close'] - rb.loc[:, 'close'].rolling(window=5).mean()) / \
             rb.loc[:, 'close'].rolling(window=5).std()
    nan = close.isnull()
    rb = rb.loc[~nan, :]
    close = close[~nan]
    return np.array(rb), np.array(close)


class CSVFileDataIter:
    def __init__(self, csv_file, seq_len, batch_size, future_size):
        self._rb, self._close = read_csv(csv_file)
        size = self._rb.shape[0]
        self._count = 0
        self._batch_size = batch_size
        self._future_size = future_size
        self._idx = [(start, start + seq_len) for
                     start in range(size - seq_len - self._future_size)]
        self._size = len(self._idx)
        logger.info('all sample size == %s', self._size)
        random.shuffle(self._idx)

    def next(self):
        data_batch, label_batch = [], []
        while len(data_batch) < self._batch_size:
            count = self._count % self._size
            start, end = self._idx[count]
            data_origin = self._rb[start: end]
            std = np.std(data_origin, axis=0, keepdims=True)
            mean = np.mean(data_origin, axis=0, keepdims=True)
            data = (data_origin - mean) / std
            if not np.all(np.isfinite(data)):
                logger.info('data nan')
                continue
            target = (self._close[end + self._future_size] >= self._close[end]).astype(np.int32)
            self._count += 1
            data_batch.append(data)
            label_batch.append(target)
        return data_batch, label_batch


class DataFrameDataIter:
    def __init__(self, seq_len, batch_size, future_size):
        self._rb, self._close = \
            concat_day_min('/home/daiab/machine_disk/code/quantum/database/RB_1day.csv',
                           '/home/daiab/machine_disk/code/quantum/database/RB_30min.csv',
                           seq_len,
                           4)
        size = self._rb.shape[0]
        self._count = 0
        self._batch_size = batch_size
        self._future_size = future_size
        self._idx = [(start, start + batch_size) for
                     start in range(size - batch_size)]
        self._size = len(self._idx)
        logger.info('all sample size == %s', self._size)
        random.shuffle(self._idx)

    def next(self):
        count = self._count % self._size
        start, end = self._idx[count]
        data_batch = self._rb[start: end]
        label_batch = self._close[start: end]
        self._count += 1
        return data_batch, label_batch


def train():
    data_iter = CSVFileDataIter(CSV_FILE, SEQ_LEN, BATCH_SIZE, 20)
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
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=context)
        for i in range(ITER_NUM_EPCOH):
            data, target = data_iter.next()
            data = mx.nd.array(data, context)
            target = mx.nd.array(target, context)
            # print(data.shape, target.shape)
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
