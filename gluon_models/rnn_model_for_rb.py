from mxnet import gluon
from mxnet.gluon import nn, rnn, autograd
import mxnet as mx
import math
import time
from gluon_models.data_iter import RecDataIter
import logging

MODEL_NAME = __name__
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    # datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=MODEL_NAME + '.log',
                    filemode='w')
logger = logging.getLogger(__name__)

BATCH_SIZE = 512
SEQ_LEN = 30
PREDICT_LEN = 1
NUM_LAYERS = 5
HIDDEN_UNITS = 380
# FC_NUM_OUTPUT = 16
NUM_CLASSES = 20
# INPUT_FC_NUM_OUPUT = 16
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
ATTN_LENGTH = 10
DROPOUT = 0.1
EPOCH = 60
# EXAMPLES = 5e6
EXAMPLES = 4030508  # for 20 classes
ITER_NUM_EPCOH = int(EXAMPLES / BATCH_SIZE)
DECAY_STEP = ITER_NUM_EPCOH * EPOCH
LR = 0.04
END_LR = 0.0002
INPUT_SIZE = 7
CELL_TYPE = 'rnn_tanh'
TRAIN_DATA_PATH = '/home/daiab/machine_disk/code/quantum/get_db_data/ohlcvr_ratio_norm.rec'
# INFER_DATA_PATH = ''
RESTORE_PATH = './model_save/%s.params' % MODEL_NAME
# infer
INFER_SIZE = 10


class RNNClsModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

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


LOG_INTERVAL = 40
data_iter = RecDataIter(TRAIN_DATA_PATH, BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
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


def train():
    for epoch in range(EPOCH):
        total_loss = 0.0
        total_acc = 0.0
        start_time = time.time()
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
