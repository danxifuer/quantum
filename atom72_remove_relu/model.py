import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from rnn_config import *
# import numpy as np


def _single_cell(num_units, cell_type, forget_bias=1.0, residual_connection=False):
    if cell_type == 'GRU':
        cell = tf.contrib.rnn.GRUCell(num_units)
    elif cell_type == 'LSTM':
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units,
                                            forget_bias=forget_bias)
    elif cell_type == 'RNN':
        cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units)
    elif cell_type == 'GridLSTM':
        cell = tf.contrib.rnn.GridLSTMCell(num_units=num_units, num_frequency_blocks=[1, 2, 3])
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


def get_model(batch_data, batch_label, is_train=True):
    cell_list = []
    for i in range(NUM_LAYERS):
        residual_connection = i >= NUM_LAYERS - NUM_RESIDUAL_LAYERS
        cell_list.append(_single_cell(HIDDEN_UNITS,
                                      CELL_TYPE,
                                      residual_connection=residual_connection))
    multi_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    batch_data = fully_connected(batch_data,
                                 num_outputs=INPUT_FC_NUM_OUPUT,
                                 activation_fn=None)
    output, state = tf.nn.dynamic_rnn(multi_cell,
                                      inputs=batch_data,
                                      dtype=tf.float32,
                                      time_major=False)
    # batch_data = tf.transpose(batch_data, [1, 0, 2])
    # batch_data = tf.unstack(batch_data, axis=0)
    # encoder_outputs, encoder_state = tf.nn.static_rnn(multi_cell,
    #                                                   inputs=batch_data,
    #                                                   dtype=tf.float32)
    # output = encoder_outputs[-PREDICT_LEN:]
    # output = tf.stack(output, axis=0)
    # output = tf.transpose(output, [1, 0, 2])
    output = output[:, -PREDICT_LEN:, :]
    print('lstm output shape: %s' % output.get_shape())
    fc_output_0 = fully_connected(inputs=tf.reshape(output, shape=(-1, HIDDEN_UNITS)),
                                  num_outputs=FC_NUM_OUTPUT,
                                  normalizer_fn=tf.contrib.layers.batch_norm)
    logits = fully_connected(fc_output_0,
                             num_outputs=2,
                             activation_fn=None)

    softmax_op = tf.nn.softmax(logits)
    if not is_train:
        return softmax_op
    # logits = tf.clip_by_value(logits, 1e-8, 0.95)
    reshaped_label = tf.reshape(batch_label, shape=(-1,))
    one_hot_label = tf.one_hot(reshaped_label, depth=2)
    print('one_hot_label shape: %s' % one_hot_label.shape)
    acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1), reshaped_label), tf.int64))
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #     labels=tf.multiply(one_hot_label, np.array([[0.97, 1.0]])),
    #     logits=logits)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #     labels=one_hot_label,
    #     logits=tf.multiply(logits, np.array([[1.0, 0.95]])))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_label,
                                                            logits=logits)
    # ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshaped_label, logits=logits)
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
