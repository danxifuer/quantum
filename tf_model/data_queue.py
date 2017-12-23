import tensorflow as tf
import logging
from functools import partial

logger = logging.getLogger(__name__)


def read_tf_records(tf_records, batch_size, capacity=300):
    if not isinstance(tf_records, (tuple, list)):
        tf_records = [tf_records]
    filename_queue = tf.train.string_input_producer(tf_records,
                                                    name='string_input')
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'label': tf.FixedLenFeature([], tf.int64),
                                                 'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, (30, 7))
    label = features['label']
    return tf.train.batch([data, label],
                          num_threads=8,
                          batch_size=batch_size,
                          capacity=capacity,
                          enqueue_many=False)


def test(dataset):
    iterator = dataset.make_initializable_iterator()
    x, y, num = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())
    ret = sess.run(x)
    print('==================')
    print(ret.shape)
    print('==================')
    sess.close()
    exit()


def get_padded_dataset(X, Y, batch_size, x_eo_id=1000, buffer_size=10):
    X = [d.T for d in X]

    def generator(x, y):
        for i in range(len(x)):
            yield x[i], y[i]

    dataset = tf.data.Dataset.from_generator(partial(generator, X, Y),
                                             (tf.float32, tf.int32),
                                             (tf.TensorShape([5, None]), tf.TensorShape([])))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(
        lambda x, y: (
            x, y, tf.shape(x)[1]
        )).prefetch(10)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([5, None]),
            tf.TensorShape([]),
            tf.TensorShape([])),
        padding_values=(
            float(x_eo_id),
            0,
            0))
    return dataset


def _unit_read_records():
    iter = tf.python_io.tf_record_iterator(
        '/home/daiab/machine_disk/code/quantum/get_db_data/ohlcvr_ratio_norm.records')
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
