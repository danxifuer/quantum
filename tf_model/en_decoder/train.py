import os

import tensorflow as tf

from data_process import get_train_data_iter
from model import get_model
from rnn_config import EPOCH, BATCH_SIZE, PREDICT_LEN, SEQ_LEN, INPUT_SIZE, RESTORE_PATH

data_p = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, SEQ_LEN, INPUT_SIZE))
label_p = tf.placeholder(dtype=tf.int64, shape=(BATCH_SIZE, PREDICT_LEN))
update, loss, acc, lr = get_model(data_p, label_p)
data_iter = get_train_data_iter()
path = os.path.dirname(RESTORE_PATH)
if not os.path.exists(path):
    os.makedirs(path)

iter_num = 0
with tf.Session() as sess:
    saver = tf.train.Saver()
    # saver.restore(sess, RESTORE_PATH)
    epoch = 0
    sess.run(tf.global_variables_initializer())
    total_loss = 0
    total_acc = 0
    while True:
        try:
            batch_data, batch_label = data_iter.next()
        except StopIteration:
            data_iter.reset()
            epoch += 1
            print('epoch == %d' % epoch)
            if epoch >= 5:
                print('model_save')
                saver.save(sess, save_path=RESTORE_PATH)
            if epoch >= EPOCH:
                break
            batch_data, batch_label = data_iter.next()
        _, loss_value, acc_value, lr_value = sess.run([update, loss, acc, lr],
                                                      feed_dict={data_p: batch_data,
                                                                 label_p: batch_label})
        total_loss += loss_value
        total_acc += acc_value
        if iter_num % 20 == 0:
            print('{}#{}; loss: {}; acc: {}; lr: {}'.format(epoch,
                                                            iter_num, total_loss / 20,
                                                            total_acc / (20 * BATCH_SIZE * PREDICT_LEN),
                                                            lr_value))
            total_acc = total_loss = 0
        iter_num += 1