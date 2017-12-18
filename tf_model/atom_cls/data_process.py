import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def read_tf_records(tfrecords, batch_size, capacity=300):
    if not isinstance(tfrecords, (tuple, list)):
        tfrecords = [tfrecords]
    filename_queue = tf.train.string_input_producer(tfrecords,
                                                    name='string_input')
    reader = tf.TFRecordReader()
    _, serilized_example = reader.read(filename_queue)
    # TODO: remove hard code 'data' and 'label'
    features = tf.parse_single_example(serilized_example,
                                       features={'label': tf.FixedLenFeature([], tf.int64),
                                                 'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, (30, 7))
    label = features['label']
    data_batch, label_batch = tf.train.batch([data, label],
                                             num_threads=8,
                                             batch_size=batch_size,
                                             capacity=capacity,
                                             enqueue_many=False)
    return data_batch, label_batch


def _unit_read_records():
    iter = tf.python_io.tf_record_iterator('/home/daiab/machine_disk/code/quantum/get_db_data/ohlcvr_ratio_norm.records')
    print(next(iter))


if __name__ == '__main__':
    # _unit_read_records()
    data_batch, label_batch = read_tf_records(
        '/home/daiab/machine_disk/code/quantum/get_db_data/ohlcvr_ratio_norm.records', 10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(sess.run(label_batch))
        coord.request_stop()
        coord.join(threads)
