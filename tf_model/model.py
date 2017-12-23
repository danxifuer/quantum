import logging
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from csv_data.read_csv_data import get_simple_data, get_concat_day_min_not_aligned
from tf_model.data_queue import get_padded_dataset

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    filename='./train.log',
                    filemode='w')
logger = logging.getLogger(__name__)

'''
def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
    """Create attention mechanism based on the attention_option."""
    # Mechanism
    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism



cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=num_units,
            alignment_history=alignment_history,
            name="attention")
'''

INPUT_SIZE = 5
BATCH_SIZE = 500
NUM_LAYERS = 5
HIDDEN_UNITS = 128
NUM_CLASSES = 2
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
ATTN_LENGTH = 10
DROPOUT_KEEP = 0.9
EPOCH = 30
EXAMPLES = 220000
DECAY_STEP = int(EXAMPLES / BATCH_SIZE) * EPOCH
LR = 0.04
END_LR = 0.0002
CELL_TYPE = 'RNN'
TRAIN_DAY_DATA = '/home/daiab/machine_disk/code/quantum/database/RB_1day.csv'
TRAIN_MIN_DATA = '/home/daiab/machine_disk/code/quantum/database/RB_30min.csv'
RESTORE_PATH = './model_save'
INFER_SIZE = 10


def _single_cell(num_units, cell_type, forget_bias=1.0, residual_connection=False):
    if cell_type == 'GRU':
        cell = tf.contrib.rnn.GRUCell(num_units)
    elif cell_type == 'LSTM':
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units,
                                            forget_bias=forget_bias)
    elif cell_type == 'RNN':
        cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units)
    elif cell_type == 'GridLSTM':
        cell = tf.contrib.rnn.GridLSTMCell(num_units=num_units,
                                           num_frequency_blocks=[1, 2, 3])
    elif cell_type == 'LayerNormLSTM':
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            dropout_keep_prob=1.0,
            forget_bias=forget_bias,
            layer_norm=True)
    else:
        raise Exception
    if residual_connection:
        cell = tf.contrib.rnn.ResidualWrapper(cell)
    return cell


def _gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    return clipped_gradients


def get_model(batch_data, batch_label, sequence_len, is_train=True):
    cell_list = []
    for i in range(NUM_LAYERS):
        residual_connection = i >= NUM_LAYERS - NUM_RESIDUAL_LAYERS
        cell = _single_cell(HIDDEN_UNITS,
                            CELL_TYPE,
                            residual_connection=residual_connection)
        # if i == 0:
        #     cell = tf.contrib.rnn.AttentionCellWrapper(cell,
        #                                                attn_length=ATTN_LENGTH,
        #                                                state_is_tuple=True)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=DROPOUT_KEEP)
        cell_list.append(cell)
    multi_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    output, state = tf.nn.dynamic_rnn(multi_cell,
                                      inputs=batch_data,
                                      sequence_length=sequence_len,
                                      dtype=tf.float32,
                                      time_major=False)
    output = output[:, -1, :]
    print('lstm output shape: %s' % output.get_shape())
    # fc_output_0 = fully_connected(inputs=output_reshape,
    #                               num_outputs=FC_NUM_OUTPUT,
    #                               normalizer_fn=tf.contrib.layers.batch_norm)
    logits = fully_connected(output,
                             num_outputs=NUM_CLASSES,
                             activation_fn=None)
    softmax_op = tf.nn.softmax(logits)
    if not is_train:
        return softmax_op
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), batch_label), tf.float32))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_label, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    trainable_vars = tf.trainable_variables()
    gradients = tf.gradients(loss, trainable_vars)  # ,
    clipped_gradients = _gradient_clip(gradients, max_gradient_norm=5.0)
    global_step = tf.Variable(0, trainable=False)
    warm_up_factor = 0.9
    warm_up_steps = 200
    learning_rate_warmup_steps = 200
    inv_decay = warm_up_factor ** (tf.to_float(warm_up_steps - global_step))
    learning_rate = LR
    warm_up_learning_rate = tf.cond(global_step < learning_rate_warmup_steps,
                                    lambda: inv_decay * learning_rate,
                                    lambda: learning_rate)
    lr = tf.train.polynomial_decay(learning_rate=warm_up_learning_rate,
                                   global_step=global_step,
                                   end_learning_rate=END_LR,
                                   decay_steps=DECAY_STEP,
                                   power=0.6)
    opt = tf.train.MomentumOptimizer(lr, 0.9)
    # opt = tf.train.AdamOptimizer(lr)
    update = opt.apply_gradients(zip(clipped_gradients, trainable_vars), global_step=global_step)
    return update, loss, acc, lr, softmax_op


class Infer:
    def __init__(self):
        self.data_iter = None  # read_tf_records(TRAIN_DATA_PATH, BATCH_SIZE)

    def validate(self, sess, softmax_op):
        filter_prob = []
        for i in range(100):
            data, label = self.data_iter.next()
            softmax_output = sess.run(softmax_op)
            real = label[-1][-1]
            # logger.info(real)
            if softmax_output[-1][1] > 0.6:  # up
                filter_prob.append((softmax_output[-1][1], 1, real))
            elif softmax_output[-1][0] > 0.6:  # down
                filter_prob.append((softmax_output[-1][0], 0, real))

        filter_prob = sorted(filter_prob, key=lambda x: -x[0])
        if len(filter_prob) > 20:
            filter_prob = filter_prob[:20]
        logger.info('filter_prob >>>>>>')
        logger.info(filter_prob[:5])
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
        logger.info('## up == (%s, %s) down == (%s, %s)' % (up_total,
                                                            up_right / (up_total + 1e-8),
                                                            down_total,
                                                            down_right / (down_total + 1e-8)))


def train():
    src, target = get_concat_day_min_not_aligned(TRAIN_DAY_DATA, TRAIN_MIN_DATA, 20)
    # src, target = min_data('/home/daiab/machine_disk/code/quantum/database/RB_min.csv',
    #                        500, 40)

    src = [d.values for d in src]
    sample_num = len(src)
    target = np.array(target, dtype=np.int32)
    tmp_data = list(zip(src, target))
    random.shuffle(tmp_data)
    src, target = list(zip(*tmp_data))
    dataset = get_padded_dataset(src, target, BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    src, target, X_padded_len = iterator.get_next()
    src = tf.transpose(src, (0, 2, 1))

    update, loss, acc, lr, softmax_op = get_model(src, target, X_padded_len)
    log_freq = 20
    # infer_class = Infer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        epoch = 0
        # saver.restore(sess, RESTORE_PATH)
        total_loss = 0
        total_acc = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(EPOCH):
            for iter_num in range(sample_num - BATCH_SIZE):
                try:
                    _, loss_value, acc_value, lr_value = \
                        sess.run([update, loss, acc, lr])
                except tf.errors.OutOfRangeError:
                    logger.info('out of range %s', iter_num)
                    sess.run(iterator.initializer)
                total_loss += loss_value
                total_acc += acc_value
                if iter_num % log_freq == 0:
                    logger.info('{}#{}; loss: {:.6f}; acc: {:.6f}; '
                                'lr: {:.8f}'.format(epoch,
                                                    iter_num,
                                                    total_loss / log_freq,
                                                    total_acc / log_freq,
                                                    lr_value))
                    total_acc = total_loss = 0
            saver.save(sess, save_path=RESTORE_PATH)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()