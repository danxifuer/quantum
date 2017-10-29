import time
import math
import mxnet as mx
from mxnet import gluon, autograd
import model
from rnn_config import *
from dataiter import DataIter

context = mx.gpu(0)
model = model.RNNModel(mode='rnn_tanh', num_embed=INPUT_SIZE,
                       num_hidden=HIDDEN_UNITS, seq_len=SEQ_LEN,
                       num_layers=NUM_LAYERS, dropout=DROPOUT)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': LR,
                         'momentum': 0,
                         'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
LOG_INTERVAL = 40


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


data_iter = DataIter(TRAIN_DATA_PATH, BATCH_SIZE)
CLIP = 0.2


def train():
    for epoch in range(EPOCH):
        total_loss = 0.0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=context)
        for i in range(ITER_NUM_EPCOH):
            data, target = data_iter.next()
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
                print('[Epoch %d Batch %d] loss %.2f, ppl %.2f' % (
                    epoch, i, cur_loss, math.exp(cur_loss)))
                total_loss = 0.0
        model.collect_params().save(RESTORE_PATH)


if __name__ == '__main__':
    train()
