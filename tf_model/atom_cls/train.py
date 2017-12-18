from data_process import read_tf_records
from rnn_config import BATCH_SIZE, PREDICT_LEN, EPOCH, ITER_NUM_EPCOH, \
    TRAIN_DATA_PATH, RESTORE_PATH
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
        self.data_iter = read_tf_records(TRAIN_DATA_PATH, BATCH_SIZE)

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


data_batch, label_batch = read_tf_records(TRAIN_DATA_PATH, BATCH_SIZE)
update, loss, acc, lr, softmax_op = get_model(data_batch, label_batch)
LOG_RATE = 40
# infer_class = Infer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    epoch = 0
    # saver.restore(sess, RESTORE_PATH)
    total_loss = 0
    total_acc = 0
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(EPOCH):
        for iter_num in range(ITER_NUM_EPCOH):
            _, loss_value, acc_value, lr_value = sess.run([update, loss, acc, lr])
            total_loss += loss_value
            total_acc += acc_value
            if iter_num % LOG_RATE == 0:
                logger.info('{}#{}; loss: {:.6f}; acc: {:.6f}; lr: {:.8f}'.format(epoch,
                                                                                  iter_num,
                                                                                  total_loss / LOG_RATE,
                                                                                  total_acc / (
                                                                                  LOG_RATE * PREDICT_LEN),
                                                                                  lr_value))
                total_acc = total_loss = 0
        saver.save(sess, save_path=RESTORE_PATH)
    coord.request_stop()
    coord.join(threads)
