from photon.preprocess import Pipeline, LabelGenerator
from photon.get_data import get_ohlcvr_and_shuffle_idx
import tensorflow as tf
import logging


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def numpy_to_tf_example(ndarray_data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'data': bytes_feature(ndarray_data.tobytes()),
        'label': int64_feature(label),
    }))


def write_ohlcvr(use_days,
                 rec_name,
                 from_date='2008-01-01',
                 end_date='2017-07-01',
                 remove_head_num=10,
                 compress=False,
                 test_write=False):
    dataset, idxs = get_ohlcvr_and_shuffle_idx(use_days,
                                               from_date=from_date,
                                               end_date=end_date,
                                               remove_head_num=remove_head_num,
                                               test_write=test_write)
    if compress:
        option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    else:
        option = None
    count = 0
    writer = tf.python_io.TFRecordWriter(rec_name, options=option)
    label_gen = LabelGenerator(20)
    pipe = Pipeline()
    max_label = 0
    min_label = 1000
    for idx in idxs:
        ori_data = dataset[idx[0]][idx[1]: idx[2]]
        # TODO: filter chain, norm_data, generate label
        data = ori_data[:, :-1]
        label = ori_data[-1, -1]
        new_data = pipe(data)
        if new_data is None:
            continue
        if new_data.dtype == np.float64:
            new_data = new_data.astype(np.float32)
        label = label_gen(label - 1.0)
        example = numpy_to_tf_example(new_data, label)
        writer.write(example.SerializeToString())
        count += 1
        if count % 1000 == 0:
            logging.info('write to records: %s', count)
        if label > max_label:
            max_label = label
        elif label < min_label:
            min_label = label
    logging.info('total write %s samples, max_label: %s, min_label: %s', count, max_label, min_label)
    writer.close()


def _unit_write():
    write_ohlcvr(30, rec_name='ohlcvr_ratio_norm.records', test_write=False)


if __name__ == '__main__':
    _unit_write()
