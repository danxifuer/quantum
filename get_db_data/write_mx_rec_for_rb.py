from get_db_data.preprocess import Pipeline, LabelGenerator, PipelineNoNorm
from get_db_data.get_data import get_ohlcvr_and_shuffle_idx, get_normed_ohlcvr_and_shuffled_idx
import mxnet as mx
import logging
import pandas as pd
import numpy as np


def write_from_csv(csv_file, use_days, rec_name):
    record = mx.recordio.MXRecordIO(rec_name, 'w')
    count = 0
    RB = pd.read_csv(csv_file, index_col=0)
    RB.index = pd.DatetimeIndex(RB.index)
    RB['CLOSE'] = RB['C']
    RB[['O', 'H', 'L', 'C', 'V']] = (RB[['O', 'H', 'L', 'C', 'V']] - RB[['O', 'H', 'L', 'C', 'V']].rolling(
        window=5).mean()) / RB[['O', 'H', 'L', 'C', 'V']].rolling(window=5).std()
    stock_data = RB.dropna()
    if regression:
        label_gen = lambda x: np.log1p(x)
    else:
        label_gen = LabelGenerator(class_num)
    pipe = PipelineNoNorm()
    max_label = 0
    min_label = 1000
    for idx in idxs:
        ori_data = dataset[idx[0]][idx[1]: idx[2]]
        # TODO: filter chain, norm_data, generate label
        data = ori_data[:, :-1]
        label = ori_data[:, -1]
        if np.any(label > 1.1) or np.any(label < 0.9) or not np.all(np.isfinite(label)):
            continue
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
    record.close()


def _unit_read():
    recordio = mx.recordio.MXRecordIO('ohlcvr_ratio_norm.rec', 'r')
    for _ in range(10):
        item = recordio.read()
        header, data = mx.recordio.unpack(item)
        array = np.frombuffer(data, np.float32)
        print(header.label)
        print(array)


if __name__ == '__main__':
    _unit_write('across_normed_ohlcvr')
    # _unit_read()
