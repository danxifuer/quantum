import numpy as np
import pickle
import random
from rnn_config import *
import logging
logger = logging.getLogger(__name__)


def _norm_zscore(data):
    data = data.copy()
    mean = np.mean(data[:, :4])
    std = np.std(data[:, :4])
    data[:, :4] = (data[:, :4] - mean) / std
    mean = np.mean(data[:, 4:], axis=0, keepdims=True)
    std = np.std(data[:, 4:], axis=0, keepdims=True)
    data[:, 4:] = (data[:, 4:] - mean) / std
    return data


def filter_remove_up_stop(data):
    if np.any(data[:, 2] == data[:, 3]):
        return True
    return False


def filter_nan(data):
    if np.any(np.isnan(data)):
        return True
    return False


def _norm_max_min(data):
    data = data.copy()
    min_value = np.min(data[:, :4])
    max_value = np.max(data[:, :4])
    if (max_value - min_value) < 1e-7:
        logger.info('data error')
        return None
    data[:, :4] = (data[:, :4] - min_value) / (max_value - min_value)
    col_num = data.shape[1]
    for i in range(4, col_num):
        min_value = np.min(data[:, i])
        max_value = np.max(data[:, i])
        if (max_value - min_value) < 1e-7:
            logger.info('data error')
            return None
        data[:, i] = (data[:, i] - min_value) / (max_value - min_value)
    return data


class DataIter:
    def __init__(self,
                 data,
                 seq_len,
                 batch_size,
                 predict_len,
                 is_train=True):
        self.data = data
        self.batch_size = batch_size
        self.predict_len = predict_len
        self.seq_len = seq_len
        self.is_train = is_train

        self.init = False
        remove_num = 0
        after_remove_data = []
        for d in self.data:
            if d.shape[0] > self.seq_len:
                after_remove_data.append(d)
            else:
                remove_num += 1
        del self.data
        logger.info('remove num: %d' % remove_num)
        self.data = new_data = []
        for d in after_remove_data:
            for i in range(0, d.shape[0] - self.seq_len):
                new_data.append(d[i:i + self.seq_len + 1])  # +1 is for label lenght == seq_len
        logger.info('train data set size = {}'.format(len(self.data)))
        self.curr_idx = 0
        self.label = None
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.curr_idx = 0
        if self.init:
            pack = list(zip(self.data, self.label))
            random.shuffle(pack)
            self.data, self.label = list(zip(*pack))
            return
        else:
            random.shuffle(self.data)
            self.init = True
        self.label = []
        data = []
        for d in self.data:
            if filter_remove_up_stop(d):
                continue
            if filter_nan(d):
                continue
            up_or_down = d[1:, IDX] / d[:-1, IDX]
            if filter_nan(up_or_down):
                continue
            up_or_down = up_or_down[-self.predict_len:]
            norm_data = _norm_zscore(d[:-1, :])
            if norm_data is not None:
                data.append(norm_data)
                self.label.append(np.where(up_or_down > 1, 1, 0))
        self.data = data
        self.all_sample_size = len(self.data)

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx >= (self.all_sample_size - self.batch_size):
            raise StopIteration
        data = self.data[self.curr_idx:self.curr_idx + self.batch_size]
        label = self.label[self.curr_idx:self.curr_idx + self.batch_size]
        self.curr_idx += self.batch_size
        data = np.array(data)
        if self.is_train:
            # return data + np.random.randn(*data.shape) * 0.005, np.array(label)
            return data, np.array(label)
        else:
            return data, np.array(label)


def get_train_data_iter():
    origin_data = pickle.load(open(TRAIN_DATA_PATH, 'rb'))
    # origin_data = [df.values[:, :INPUT_SIZE] for df in origin_data if df is not None]
    logger.info('all origin data num = %s ' % len(origin_data))
    # origin_data = origin_data[:100]
    return DataIter(origin_data,
                    seq_len=SEQ_LEN,
                    batch_size=BATCH_SIZE,
                    predict_len=PREDICT_LEN)


def get_infer_data_iter(batch_size=1):
    origin_data = pickle.load(open(INFER_DATA_PATH, 'rb'))
    logger.info('all origin data num = %s ' % len(origin_data))
    # origin_data = origin_data[:10]
    return DataIter(origin_data,
                    seq_len=SEQ_LEN,
                    batch_size=1,
                    predict_len=PREDICT_LEN,
                    is_train=False)



