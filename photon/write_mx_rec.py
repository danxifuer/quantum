from photon.preprocess import Pipeline, LabelGenerator
from photon.get_data import get_ohlcvr_and_shuffle_idx
import mxnet as mx
import logging
import numpy as np


def write_ohlcvr(use_days,
                 rec_name,
                 from_date='2008-01-01',
                 end_date='2017-07-01',
                 remove_head_num=10,
                 test_write=False):
    dataset, idxs = get_ohlcvr_and_shuffle_idx(use_days,
                                               from_date=from_date,
                                               end_date=end_date,
                                               remove_head_num=remove_head_num,
                                               test_write=test_write)
    record = mx.recordio.MXRecordIO(rec_name, 'w')
    count = 0
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
        header = mx.recordio.IRHeader(0, label, count, 0)
        packed_s = mx.recordio.pack(header, new_data.tobytes())
        record.write(packed_s)
        count += 1
        if count % 1000 == 0:
            logging.info('write to records: %s', count)
        if label > max_label:
            max_label = label
        elif label < min_label:
            min_label = label
    logging.info('total write %s samples, max_label: %s, min_label: %s', count, max_label, min_label)
    record.close()


def _unit_write():
    write_ohlcvr(30, rec_name='ohlcvr_ratio_norm.rec', test_write=True)


def _unit_read():
    recordio = mx.recordio.MXRecordIO('ohlcvr_ratio_norm.rec', 'r')
    for _ in range(10):
        item = recordio.read()
        header, data = mx.recordio.unpack(item)
        array = np.frombuffer(data, np.float32)
        print(header.label)
        print(array)



if __name__ == '__main__':
    _unit_write()
    # _unit_read()
