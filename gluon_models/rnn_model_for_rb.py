import logging
import math
import random
import matplotlib.pyplot as plt
import mxnet as mx
import pandas as pd
import numpy as np
import time
from queue import Queue
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import nn, rnn
from threading import Thread
from csv_data.read_csv_data import get_simple_data, get_data_ma_smooth

MODEL_NAME = __name__
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    filename=MODEL_NAME + '.log',
                    filemode='w')
logger = logging.getLogger(__name__)

BATCH_SIZE = 256
SEQ_LEN = 500
PREDICT_LEN = 40
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


constant_vars = tuple(vars().items())
for k, v in constant_vars:
    if not k.startswith('__') and k.isupper():
        logger.info('%s: %s', k, v)


class RNNClsModel(gluon.Block):
    def __init__(self, mode, num_embed, num_hidden, seq_len,
                 num_layers, dropout=0.0, **kwargs):
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

            self.seq_len = seq_len
            if isinstance(seq_len, (list, tuple)):
                self.seq_len = min(seq_len)
            self.fc = nn.Dense(NUM_CLASSES, in_units=num_hidden * self.seq_len)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        output = self.drop(output)
        size = output.shape[1]
        if size != self.seq_len:
            output = output[:, size - self.seq_len: size, :]
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
        # self._X, self._Y = get_simple_data(csv_file, seq_len, predict_len)
        self._X, self._Y = get_data_ma_smooth(csv_file, seq_len, predict_len)
        self._X = [d.values for d in self._X]
        X_Y = list(zip(self._X, self._Y))
        random.shuffle(X_Y)
        self._X, self._Y = list(zip(*X_Y))
        train_num = int(len(self._X) * 0.9)
        self._train_X = self._X[:train_num]
        self._val_X = self._X[train_num:]
        self._train_Y = self._Y[:train_num]
        self._val_Y = self._Y[train_num:]

        self._train_idx = [(start, start + batch_size) for start in range(0, len(self._train_X) - batch_size, batch_size)]
        self._val_idx = [(start, start + batch_size) for start in range(0, len(self._val_X) - batch_size, batch_size)]
        logger.info('train batch num = %s, valid batch num = %s', len(self._train_idx), len(self._val_idx))
        self._train_count = 0
        self._valid_count = 0

    def _reset(self):
        X_Y = list(zip(self._train_X, self._train_Y))
        random.shuffle(X_Y)
        self._train_X, self._train_Y = list(zip(*X_Y))

    def next(self):
        self._train_count += 1
        if self._train_count % len(self._train_idx) == 0:
            self._reset()
            raise StopIteration
        start, end = random.choice(self._train_idx)
        data_batch = self._X[start: end]
        label_batch = self._Y[start: end]
        return data_batch, label_batch

    def next_valid(self):
        self._valid_count += 1
        count = self._valid_count % len(self._val_idx)
        if count == 0:
            raise StopIteration
        start, end = self._val_idx[count]
        return self._X[start: end], self._Y[start: end]


class VarSeqLenDataIter:
    def __init__(self, csv_file, seq_len_low, seq_len_up, batch_size, predict_len):
        self._seq_len_low = seq_len_low
        self._seq_len_up = seq_len_up
        self._batch_size = batch_size
        self._predict_len = predict_len
        # self._csv_file = csv_file
        self._rb = self._read_csv(csv_file, 5)
        length = self._rb.shape[0]
        idx = list(range(seq_len_up, length - predict_len))
        random.shuffle(idx)
        self._idx_for_train = idx[:int(length * 0.9)]
        self._idx_for_valid = idx[int(length * 0.9):]
        self._seq_len_range = list(range(self._seq_len_low, self._seq_len_up))
        self._valid_count = 0
        self._valid_seq_count = 0
        self._train_count = 0
        self._iter_per_epoch = int(len(self._idx_for_train) * len(self._seq_len_range) / batch_size)
        self._valid_iter_num = int(len(self._idx_for_valid) / batch_size)
        logger.info('train batch num = %s, valid batch num = %s',
                    self._iter_per_epoch,
                    self._valid_iter_num * len(self._seq_len_range)
                    )
        self._data_queue = Queue()
        th = Thread(target=self._data_pipe)
        th.daemon = True
        th.start()

    def _data_pipe(self):
        while True:
            if self._data_queue.qsize() > 2000:
                time.sleep(0.2)
                continue
            seq_len = random.choice(self._seq_len_range)
            batch_data, batch_target = self._read_train_batch(seq_len)
            self._data_queue.put((batch_data, batch_target))

    @staticmethod
    def _read_csv(csv_file, ma_period):
        rb = pd.read_csv(csv_file, index_col=0)
        # rb.index = pd.DatetimeIndex(rb.index)
        rb = rb.rolling(ma_period).mean()
        rb = rb.dropna()
        return rb

    def _read_train_batch(self, seq_len):
        batch_data = []
        batch_target = []
        idx_set = set()
        while len(batch_data) < self._batch_size:
            i = random.choice(self._idx_for_train)
            while i in idx_set:
                i = random.choice(self._idx_for_train)
            idx_set.add(i)
            start, end = i - seq_len, i
            data = self._rb.iloc[start: end]
            target = self._rb['close']
            data = (data - data.mean()) / data.std()
            if not np.all(np.isfinite(data)):
                print('data nan')
                continue
            y = int(target.iloc[i + self._predict_len - 1] >= target.iloc[i - 1])
            batch_data.append(data.values)
            batch_target.append(y)
        return batch_data, batch_target

    def _read_valid_batch(self, seq_len):
        self._valid_count += 1
        idx = self._valid_count % self._valid_iter_num
        if idx == 0:
            raise StopIteration
        batch_data = []
        batch_target = []
        while len(batch_data) < self._batch_size:
            i = self._idx_for_valid[idx]
            start, end = i - seq_len, i
            data = self._rb.iloc[start: end]
            target = self._rb['close']
            data = (data - data.mean()) / data.std()
            if not np.all(np.isfinite(data)):
                logger.error('data nan')
                continue
            y = int(target.iloc[i + self._predict_len - 1] >= target.iloc[i - 1])
            batch_data.append(data.values)
            batch_target.append(y)
        return batch_data, batch_target

    def next(self):
        self._train_count += 1
        if self._train_count % self._iter_per_epoch == 0:
            raise StopIteration
        # seq_len = random.choice(self._seq_len_range)
        # return self._read_train_batch(seq_len)
        return self._data_queue.get(timeout=10)

    def next_valid(self):
        self._valid_seq_count += 1
        seq_len_idx = self._valid_seq_count % len(self._seq_len_range)
        if seq_len_idx == 0:
            raise StopIteration
        try:
            seq_len = self._seq_len_range[seq_len_idx]
            return self._read_valid_batch(seq_len)
        except StopIteration:
            return self.next_valid()


class TrainModel:
    def __init__(self):
        self.ctx = mx.gpu(0)
        # self.iterator = DataIter(CSV_FILE, SEQ_LEN, BATCH_SIZE, predict_len=PREDICT_LEN)
        # seq_len = SEQ_LEN
        self.iterator = VarSeqLenDataIter(CSV_FILE, SEQ_LEN, SEQ_LEN+100, BATCH_SIZE, predict_len=PREDICT_LEN)
        seq_len = (SEQ_LEN, SEQ_LEN+20)
        self.model = RNNClsModel(mode=CELL_TYPE, num_embed=INPUT_SIZE,
                                 num_hidden=HIDDEN_UNITS, seq_len=seq_len,
                                 num_layers=NUM_LAYERS, dropout=DROPOUT)
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
        self.train_record = []
        self.val_record = []
        logger.info('read data and model init over')

    def train(self):
        LOG_INTERVAL = 20
        CLIP = 0.2
        self.model.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        # model.collect_params().load(RESTORE_PATH, context)
        trainer = gluon.Trainer(self.model.collect_params(), 'sgd',
                                {'learning_rate': LR,
                                 'momentum': 0.9,
                                 'wd': 0.0})
        lr_decay = LRDecay(LR, END_LR, 0.6, DECAY_STEP)
        for epoch in range(EPOCH):
            total_loss = 0.0
            total_acc = 0.0
            hidden = self.model.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=self.ctx)
            epoch_loss = []
            epoch_acc = []
            iter_num = -1
            while True:
                iter_num += 1
                try:
                    data, target = self.iterator.next()
                except StopIteration:
                    break
                data = mx.nd.array(data, self.ctx)
                target = mx.nd.array(target, self.ctx)
                hidden = detach(hidden)
                with autograd.record():
                    output, hidden = self.model(data, hidden)
                    L = self.loss(output, target)
                    L.backward()

                grads = [i.grad(self.ctx) for i in self.model.collect_params().values()]
                # Here gradient is for the whole batch.
                # So we multiply max_norm by batch_size and bptt size to balance it.
                gluon.utils.clip_global_norm(grads, CLIP * BATCH_SIZE)

                trainer.step(BATCH_SIZE)
                total_loss += mx.nd.sum(L).asscalar()
                total_acc += mx.nd.sum(mx.nd.equal(mx.nd.argmax(output, axis=1), target)).asscalar()

                if iter_num % LOG_INTERVAL == 0:
                    cur_loss = total_loss / BATCH_SIZE / LOG_INTERVAL
                    cur_acc = total_acc / BATCH_SIZE / LOG_INTERVAL
                    logger.info('%d # %d loss %.5f, ppl %.5f, lr %.5f, acc %.5f',
                                epoch, iter_num, cur_loss, math.exp(cur_loss), trainer._optimizer.lr, cur_acc)
                    epoch_loss.append(cur_loss)
                    epoch_acc.append(cur_acc)
                    total_loss = 0.0
                    total_acc = 0.0
                trainer._optimizer.lr = lr_decay.lr
            self.model.collect_params().save(RESTORE_PATH)
            self.train_record.append((sum(epoch_loss) / len(epoch_loss),
                                      sum(epoch_acc) / len(epoch_acc)))
            self.valid()
            self.plot()

    def valid(self):
        hidden = self.model.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=self.ctx)
        logger.info('start to valid')
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        try:
            while True:
                data, target = self.iterator.next_valid()
                data = mx.nd.array(data, self.ctx)
                target = mx.nd.array(target, self.ctx)
                target = mx.nd.reshape(target, shape=(-1,))
                output, _ = self.model(data, hidden)
                L = self.loss(output, target)
                total_loss += mx.nd.sum(L).asscalar()
                total_acc += mx.nd.sum(mx.nd.equal(mx.nd.argmax(output, axis=1), target)).asscalar()
                count += BATCH_SIZE
        except StopIteration:
            cur_loss = total_loss / count
            cur_acc = total_acc / count
            logger.info('valid: loss %.5f, ppl %.5f, acc %.5f',
                        cur_loss, math.exp(cur_loss), cur_acc)
            self.val_record.append((cur_loss, cur_acc))

    def plot(self):
        train = np.array(self.train_record)
        val = np.array(self.val_record)
        x = np.arange(0, train.shape[0])

        fig, axes = plt.subplots(nrows=2, figsize=(18, 9))
        ax = axes[0]
        ax.plot(x, train[:, 0], 'o-')
        ax.plot(x, val[:, 0], '*-')
        ax.set_title("loss")

        ax = axes[1]
        ax.plot(x, train[:, 1], 'o-')
        ax.plot(x, val[:, 1], '*-')
        ax.set_title("acc")
        plt.savefig('train_val.svg', format='svg', dpi=800)


if __name__ == '__main__':
    train_model = TrainModel()
    train_model.train()
