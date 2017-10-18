from data_process import get_train_data_iter, get_infer_data_iter
from rnn_config import EPOCH, BATCH_SIZE, PREDICT_LEN, SEQ_LEN, INPUT_SIZE, RESTORE_PATH
from model import get_model
import logging
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./train.log',
                    filemode='w')
logger = logging.getLogger(__name__)


class Infer:
    def __init__(self):
        self.data_iter = get_infer_data_iter()

    def validate(self, sess, loss):
        filter_prob = []
        while True:
            try:
                data, label = self.data_iter.next()
                loss_value = sess.run(loss, feed_dict={data_p: data})
                filter_prob.append(loss_value)
            except StopIteration:
                self.data_iter.reset()
                filter_prob = sorted(filter_prob, key=lambda x: x)
                logger.info('total mean loss == %s', sum(filter_prob)/ len(filter_prob))
                if len(filter_prob) > 20:
                    filter_prob = filter_prob[:20]
                    logger.info('first 20 mean loss == %s', sum(filter_prob) / len(filter_prob))


data_p = tf.placeholder(dtype=tf.float32, shape=(None, SEQ_LEN, INPUT_SIZE))
label_p = tf.placeholder(dtype=tf.int64, shape=(None, PREDICT_LEN))
update, loss, lr = get_model(data_p, label_p)
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
    while True:
        try:
            batch_data, batch_label = data_iter.next()
        except StopIteration:
            infer_class.validate(sess, loss)
            data_iter.reset()
            epoch += 1
            logger.info('epoch == %d' % epoch)
            if epoch >= 5:
                saver.save(sess, save_path=RESTORE_PATH)
            if epoch >= EPOCH:
                break
            batch_data, batch_label = data_iter.next()
        _, loss_value, lr_value = sess.run([update, loss, lr],
                                           feed_dict={data_p: batch_data,
                                                      label_p: batch_label})
        total_loss += loss_value
        if iter_num % LOG_RATE == 0:
            logger.info('{}#{}; loss: {}; lr: {}'.format(epoch,
                                                         iter_num, total_loss / LOG_RATE,
                                                         lr_value))
        iter_num += 1
