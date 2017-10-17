from data_process import get_train_data_iter, get_infer_data_iter
import tensorflow as tf
from rnn_config import EPOCH, BATCH_SIZE, PREDICT_LEN, SEQ_LEN, INPUT_SIZE, RESTORE_PATH
from model import get_model


class Infer:
    def __init__(self):
        self.data_iter = get_infer_data_iter()

    def validate(self, sess, softmax_op):
        filter_prob = []
        while True:
            try:
                data, label = self.data_iter.next()
                softmax_output = sess.run(softmax_op, feed_dict={data_p: data})
                real = label[-1][-1]
                # print(real)
                if softmax_output[-1][1] > 0.7:  # up
                    filter_prob.append((softmax_output[-1][1], 1, real))
                elif softmax_output[-1][0] > 0.7:  # down
                    filter_prob.append((softmax_output[-1][0], 0, real))
            except StopIteration:
                self.data_iter.reset()
                filter_prob = sorted(filter_prob, key=lambda x: -x[0])
                if len(filter_prob) > 20:
                    filter_prob = filter_prob[:20]
                print('filter_prob >>>>>>')
                print(filter_prob[:5])
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
                if down_total == 0 or up_total == 0:
                    print('error and passed, down_total: %s, up_total: %s' % (down_total, up_total))
                    return
                print('## up == (%s, %s) down == (%s, %s)' % (up_total,
                                                              up_right / up_total,
                                                              down_total,
                                                              down_right / down_total))
                break


data_p = tf.placeholder(dtype=tf.float32, shape=(None, SEQ_LEN, INPUT_SIZE))
label_p = tf.placeholder(dtype=tf.int64, shape=(None, PREDICT_LEN))
update, loss, acc, lr, softmax_op = get_model(data_p, label_p)
data_iter = get_train_data_iter()

iter_num = 0
LOG_RATE = 40

infer_class = Infer()


with tf.Session() as sess:
    saver = tf.train.Saver()
    epoch = 0
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, RESTORE_PATH)
    total_loss = 0
    total_acc = 0
    while True:
        try:
            batch_data, batch_label = data_iter.next()
        except StopIteration:
            infer_class.validate(sess, softmax_op)
            data_iter.reset()
            epoch += 1
            print('epoch == %d' % epoch)
            if epoch >= 5:
                saver.save(sess, save_path=RESTORE_PATH)
            if epoch >= EPOCH:
                break
            batch_data, batch_label = data_iter.next()
        _, loss_value, acc_value, lr_value = sess.run([update, loss, acc, lr],
                                                      feed_dict={data_p: batch_data,
                                                                 label_p: batch_label})
        total_loss += loss_value
        total_acc += acc_value
        if iter_num % LOG_RATE == 0:
            print('{}#{}; loss: {}; acc: {}; lr: {}'.format(epoch,
                                                            iter_num, total_loss / LOG_RATE,
                                                            total_acc / (LOG_RATE * BATCH_SIZE * PREDICT_LEN),
                                                            lr_value))
            total_acc = total_loss = 0
        iter_num += 1
