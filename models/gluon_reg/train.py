import time
import math
import mxnet as mx
from mxnet import gluon, autograd
import model
import logging
from rnn_config import *
from dataiter import DataIter

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    # datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./train.log',
                    filemode='w')
logger = logging.getLogger(__name__)


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
data_iter = DataIter(TRAIN_DATA_PATH, BATCH_SIZE)
CLIP = 0.2
context = mx.gpu(0)
model = model.RNNModel(mode=CELL_TYPE, num_embed=INPUT_SIZE,
                       num_hidden=HIDDEN_UNITS, seq_len=SEQ_LEN,
                       num_layers=NUM_LAYERS, dropout=DROPOUT)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
# model.collect_params().load(RESTORE_PATH, context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': LR,
                         'momentum': 0.9,
                         'wd': 0.00005})
loss = gluon.loss.L2Loss()
lr_decay = LRDecay(LR, END_LR, 0.6, DECAY_STEP)


def train():
    for epoch in range(EPOCH):
        total_loss = 0.0
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

            if i % LOG_INTERVAL == 0:
                cur_loss = total_loss / BATCH_SIZE / LOG_INTERVAL
                logger.info('%d # %d loss %.5f, ppl %.5f, lr %.5f',
                            epoch, i, cur_loss, math.exp(cur_loss), trainer._optimizer.lr)
                total_loss = 0.0
            trainer._optimizer.lr = lr_decay.lr
        model.collect_params().save(RESTORE_PATH)


if __name__ == '__main__':
    train()
