from photon.db_access import *
from photon.aux import Pipeline, LabelGenerator
from random import shuffle
import tensorflow as tf
import pickle


def get_ohlcvr_and_shuffle_idx(use_days, remove_head_num=10):
    code_list = get_code(greater_days=200)
    all_data = []
    # TODO: delete this
    # code_list = code_list[:3]

    for code in code_list:
        tmp = get_ohlcv_future_ret(code)
        if len(tmp) <= (remove_head_num + use_days + 5):  # +5 is avoid error
            continue
        all_data.append(tmp[remove_head_num:])
    idx_list = []
    for i, d in enumerate(all_data):
        for s in range(len(d) - use_days):
            idx_list.append((i, s, s + use_days))  # i th stock, start, end,
    shuffle(idx_list)
    return all_data, idx_list


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


def write_ohlcvr(use_days, rec_name, remove_head_num=10, compress=False):
    dataset, idxs = get_ohlcvr_and_shuffle_idx(use_days, remove_head_num)
    if compress:
        option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    else:
        option = None
    count = 0
    writer = tf.python_io.TFRecordWriter(rec_name, options=option)
    label_gen = LabelGenerator(40)
    pipe = Pipeline()
    for idx in idxs:
        ori_data = dataset[idx[0]][idx[1]: idx[2]]
        # TODO: filter chain, norm_data, generate label
        data = ori_data[:, :-1]
        label = ori_data[-1, -1]
        new_data = pipe(data)
        if new_data is None:
            continue
        label = label_gen(label)
        example = numpy_to_tf_example(new_data, label)
        writer.write(example.SerializeToString())
        count += 1
        if count % 1000 == 0:
            print(count)
    writer.close()


def _unit_write():
    write_ohlcvr(30, 'ohlcvr_ratio_norm.records')


if __name__ == '__main__':
    _unit_write()
