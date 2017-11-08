import numpy as np
import math
import mxnet as mx
from mxnet import gluon, autograd
import model
import logging
from rnn_config import *
import matplotlib.pyplot as plt
from dataiter import DataIter, InferDataIter

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
CLIP = 0.1


# model.collect_params().load(RESTORE_PATH, context)


class ModelTrain:
    def __init__(self):
        self.context = mx.gpu(0)
        self.model = model.RNNModel(mode=CELL_TYPE, num_embed=INPUT_SIZE,
                                    num_hidden=HIDDEN_UNITS, seq_len=SEQ_LEN,
                                    num_layers=NUM_LAYERS, dropout=DROPOUT)
        self._model_init()
        self.trainer = gluon.Trainer(self.model.collect_params(), 'sgd',
                                     {'learning_rate': LR,
                                      'momentum': 0.9,
                                      'wd': 0.0001})
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
        self.lr_decay = LRDecay(LR, END_LR, 0.6, DECAY_STEP)
        self.data_iter = DataIter(TRAIN_DATA_PATH, BATCH_SIZE)
        self.infer_iter = InferDataIter(INFER_DATA_PATH, BATCH_SIZE)
        self.train_record = []
        self.val_record = []

    def _model_init(self):
        self.model.collect_params().initialize(mx.init.Xavier(), ctx=self.context)

    def run(self):
        hidden = self.model.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=self.context)
        for epoch in range(EPOCH):
            tmp_loss = 0.0
            tmp_acc = 0.0
            epoch_loss = []
            epoch_acc = []
            total = BATCH_SIZE * PREDICT_LEN * LOG_INTERVAL
            for i in range(ITER_NUM_EPCOH):
                data, target = self.data_iter.next()
                data = mx.nd.array(data, self.context)
                target = mx.nd.array(target, self.context)
                target = mx.nd.reshape(target, shape=(-1,))
                hidden = detach(hidden)
                with autograd.record():
                    output, hidden = self.model(data, hidden)
                    L = self.loss(output, target)
                    L.backward()

                grads = [i.grad(self.context) for i in self.model.collect_params().values()]
                # Here gradient is for the whole batch.
                # So we multiply max_norm by batch_size and bptt size to balance it.
                global_norm = gluon.utils.clip_global_norm(grads, CLIP * BATCH_SIZE)

                self.trainer.step(BATCH_SIZE)
                tmp_loss += mx.nd.sum(L).asscalar()
                tmp_acc += mx.nd.sum(mx.nd.equal(mx.nd.argmax(output, axis=1), target)).asscalar()

                if i % LOG_INTERVAL == 0:
                    cur_loss = tmp_loss / total
                    cur_acc = tmp_acc / total
                    logger.info('%d # %d loss %.5f, ppl %.5f, lr %.5f, acc %.5f',
                                epoch, i, cur_loss, math.exp(cur_loss), self.trainer._optimizer.lr, cur_acc)
                    epoch_loss.append(cur_loss)
                    epoch_acc.append(cur_acc)
                    tmp_loss = 0.0
                    tmp_acc = 0.0
                self.trainer._optimizer.lr = self.lr_decay.lr
            self.model.collect_params().save(RESTORE_PATH)
            self.valid(hidden)
            self.train_record.append((sum(epoch_loss) / len(epoch_loss),
                                      sum(epoch_acc) / len(epoch_acc)))
            self.plot()

    def valid(self, hidden):
        logger.info('start to valid')
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        try:
            while True:
                data, target = self.infer_iter.next()
                data = mx.nd.array(data, self.context)
                target = mx.nd.array(target, self.context)
                target = mx.nd.reshape(target, shape=(-1,))
                output, _ = self.model(data, hidden)
                L = self.loss(output, target)
                total_loss += mx.nd.sum(L).asscalar()
                total_acc += mx.nd.sum(mx.nd.equal(mx.nd.argmax(output, axis=1), target)).asscalar()
                count += BATCH_SIZE
        except:
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
        plt.show()


if __name__ == '__main__':
    ModelTrain().run()
