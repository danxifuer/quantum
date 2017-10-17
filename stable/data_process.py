import numpy as np
import pickle
import random
from rnn_config import *


def _norm_zscore(data):
    mean = np.mean(data[:, :4])
    std = np.std(data[:, :4])
    data[:, :4] = (data[:, :4] - mean) / std
    mean = np.mean(data[:, 4:], axis=0, keepdims=True)
    std = np.std(data[:, 4:], axis=0, keepdims=True)
    data[:, 4:] = (data[:, 4:] - mean) / std
    # if np.any(data > 20):
    #     print(data)
    #     exit()
    # if np.min(data) < -5.0:
    #     print(np.min(data))
    return data


def _norm_max_min(data):
    data = data.copy()
    min_value = np.min(data[:, :4])
    max_value = np.max(data[:, :4])
    if (max_value - min_value) < 1e-7:
        print('data error')
        return None
    data[:, :4] = (data[:, :4] - min_value) / (max_value - min_value)
    col_num = data.shape[1]
    for i in range(4, col_num):
        min_value = np.min(data[:, i])
        max_value = np.max(data[:, i])
        if (max_value - min_value) < 1e-7:
            print('data error')
            return None
        data[:, i] = (data[:, i] - min_value) / (max_value - min_value)
    return data


class DataIter:
    def __init__(self,
                 data,
                 seq_len,
                 batch_size,
                 predict_len):
        self.data = data
        self.batch_size = batch_size
        self.predict_len = predict_len
        self.seq_len = seq_len

        self.init = False
        remove_num = 0
        after_remove_data = []
        for d in self.data:
            if d.shape[0] > self.seq_len:
                after_remove_data.append(d)
            else:
                remove_num += 1
        del self.data
        print('remove num: %d' % remove_num)
        new_data = []
        for d in after_remove_data:
            for i in range(0, d.shape[0] - self.seq_len):
                new_data.append(d[i:i + self.seq_len + 1])  # +1 is for label lenght == seq_len
        self.data = np.array(new_data)
        print('train data set size = {}'.format(self.data.shape))
        self.all_sample_size = self.data.shape[0]
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
            np.random.shuffle(self.data)
            self.init = True
        self.label = []
        data = []
        for i in range(self.all_sample_size):
            d = self.data[i]
            if np.any(np.isnan(d)):
                print('data here have nan')
                continue
            up_or_down = d[1:, IDX] / d[:-1, IDX]
            if np.any(np.isnan(up_or_down)):
                print('here have nan')
                continue
            up_or_down = up_or_down[-self.predict_len:]
            norm_data = _norm_max_min(d[:-1, :])
            if norm_data is not None:
                data.append(norm_data)
                self.label.append(np.where(up_or_down > 1, 1, 0))
        self.data = data

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx >= (self.all_sample_size - self.batch_size):
            raise StopIteration
        data = self.data[self.curr_idx:self.curr_idx + self.batch_size]
        label = self.label[self.curr_idx:self.curr_idx + self.batch_size]
        self.curr_idx += self.batch_size
        return np.array(data), np.array(label)


def get_data_iter():
    origin_data = pickle.load(open(DATA_PATH, 'rb'))
    # origin_data = [df.values[:, :INPUT_SIZE] for df in origin_data if df is not None]
    print('all origin data num = %s ' % len(origin_data))
    origin_data = origin_data[:100]
    return DataIter(origin_data,
                    seq_len=SEQ_LEN,
                    batch_size=BATCH_SIZE,
                    predict_len=PREDICT_LEN)
