import logging
import math
import random
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
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


constant_vars = tuple(vars().items())
for k, v in constant_vars:
    if not k.startswith('__') and k.isupper():
        logger.info('%s: %s', k, v)


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
        # self._X, self._Y = get_simple_data(csv_file, seq_len, predict_len)
        self._X, self._Y = get_data_ma_smooth(csv_file, seq_len, predict_len)
        self._X = [d.values for d in self._X]
        size = len(self._X)
        self._count = 0
        self._batch_size = batch_size
        self._idx = [(start, start + batch_size) for start in range(0, size - batch_size, batch_size)]
        self._size = len(self._idx)
        random.shuffle(self._idx)
        train_num = int(self._size * 0.9)
        self._train_idx = self._idx[:train_num]
        self._val_idx = self._idx[train_num:]
        logger.info('train batch num = %s, valid batch num = %s', len(self._train_idx), len(self._val_idx))
        self._train_count = 0
        self._valid_count = 0

    def next(self):
        self._train_count += 1
        if self._train_count % len(self._train_idx) == 0:
            raise StopIteration
        self._train_count += 1
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


class TrainModel:
    def __init__(self):
        self.iterator = DataIter(CSV_FILE, SEQ_LEN, BATCH_SIZE, predict_len=40)
        self.ctx = mx.gpu(0)
        print('read data over')
        self.model = RNNClsModel(mode=CELL_TYPE, num_embed=INPUT_SIZE,
                                 num_hidden=HIDDEN_UNITS, seq_len=SEQ_LEN,
                                 num_layers=NUM_LAYERS, dropout=DROPOUT)
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
        self.train_record = []
        self.val_record = []

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

    def valid(self):
        hidden = self.model.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=self.ctx)
        logger.info('start to valid')
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        try:
            while True:
                data, target = self.iterator.next_valid()
                data = mx.nd.array(data, self.context)
                target = mx.nd.array(target, self.context)
                target = mx.nd.reshape(target, shape=(-1,))
                output, _ = self.model(data, hidden)
                L = self.loss(output, target)
                total_loss += mx.nd.sum(L).asscalar()
                total_acc += mx.nd.sum(mx.nd.equal(mx.nd.argmax(output, axis=1), target)).asscalar()
                count += BATCH_SIZE
        except StopIteration:
            cur_loss = total_loss / count / PREDICT_LEN
            cur_acc = total_acc / count / PREDICT_LEN
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
