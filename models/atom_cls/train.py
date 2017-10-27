from data_process import read_tf_records
from rnn_config import BATCH_SIZE, PREDICT_LEN, DECAY_STEP
from model import get_model
import logging
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    # datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./train.log',
                    filemode='w')
logger = logging.getLogger(__name__)


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
                # logger.info(real)
                if softmax_output[-1][1] > 0.6:  # up
                    filter_prob.append((softmax_output[-1][1], 1, real))
                elif softmax_output[-1][0] > 0.6:  # down
                    filter_prob.append((softmax_output[-1][0], 0, real))
            except StopIteration:
                self.data_iter.reset()
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
                break


data_batch, label_batch = read_tf_records(
    '/home/daiab/machine_disk/code/quantum/photon/ohlcvr_ratio_norm.records', BATCH_SIZE)
update, loss, acc, lr, softmax_op = get_model(data_batch, label_batch)
iter_num = 0
LOG_RATE = 40

# infer_class = Infer()
with tf.Session() as sess:
    saver = tf.train.Saver()
    epoch = 0
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, RESTORE_PATH)
    total_loss = 0
    total_acc = 0
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    while True:

        _, loss_value, acc_value, lr_value = sess.run([update, loss, acc, lr])
        total_loss += loss_value
        total_acc += acc_value
        if iter_num % LOG_RATE == 0:
            logger.info('{}#{}; loss: {:.6f}; acc: {:.6f}; lr: {:.8f}'.format(epoch,
                                                                              iter_num, total_loss / LOG_RATE,
                                                                              total_acc / (
                                                                              LOG_RATE * BATCH_SIZE * PREDICT_LEN),
                                                                              lr_value))
            total_acc = total_loss = 0
        if iter_num > DECAY_STEP:
            break
        iter_num += 1
    coord.request_stop()
    coord.join(threads)
