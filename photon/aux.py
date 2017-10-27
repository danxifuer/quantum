# import tensorflow as tf
import numpy as np
import logging


def _clip_by_std(mean, std, data):
    mask = np.where(data > 3 * std)
    data[mask] = 3 * std
    mask = np.where(data < - 3 * std)
    data[mask] = - 3 * std
    return data


def _zscore(data):
    std = np.std(data, axis=0, keepdims=True)
    mean = np.mean(data, axis=0, keepdims=True)
    return (data - mean) / std


class Filter:
    def __call__(self, data):
        raise NotImplementedError


class Norm:
    def __call__(self, data, **kwargs):
        raise NotImplementedError


class NaNFilter(Filter):
    def __call__(self, data):
        if np.any(np.isnan(data)):
            return None
        return data


class ContinueUpFilter(Filter):
    def __init__(self, high_idx, low_idx):
        self.high_idx = high_idx
        self.low_idx = low_idx

    def __call__(self, data):
        minus = data[self.high_idx] - data[self.low_idx]
        if np.any(np.abs(minus) < 1e-5):
            return None
        return data


class ZScoreNorm(Norm):
    def __call__(self, data, clip=True):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        std = np.std(data, axis=0, keepdims=True)
        mean = np.mean(data, axis=0, keepdims=True)
        ret = (data - mean) / std
        if clip:
            ret = _clip_by_std(mean, std, ret)
        return ret


class MaxMinNorm(Norm):
    def __call__(self, data, **kwargs):
        raise NotImplementedError


class RatioNorm(Norm):
    def __init__(self, idx_map, use_log1p=True):
        self.idx_map = idx_map
        if use_log1p:
            self.func = np.log1p
        else:
            self.func = lambda x: x

    def __call__(self, data, **kwargs):
        ret = np.empty(shape=(data.shape[0], 7), dtype=np.float32)
        ret[:, 0] = self.func(data[:, self.idx_map['close']] / data[:, self.idx_map['open']])
        ret[:, 1] = self.func(data[:, self.idx_map['high']] / data[:, self.idx_map['open']])
        ret[:, 2] = self.func(data[:, self.idx_map['low']] / data[:, self.idx_map['open']])
        ret[:, 3] = self.func(data[:, self.idx_map['open']] / data[:, self.idx_map['close']])
        ret[:, 4] = self.func(data[:, self.idx_map['high']] / data[:, self.idx_map['close']])
        ret[:, 5] = self.func(data[:, self.idx_map['low']] / data[:, self.idx_map['close']])
        ret[:, 6] = _zscore(data[:, self.idx_map['volume']])
        return ret


class LabelGenerator:
    def __init__(self, num_classes):
        assert num_classes % 2 == 0, 'num_classes % 2 != 0'
        self.num_classes = num_classes
        self._threshold = int(self.num_classes / 2)

    def __call__(self, returns):
        value = int(returns * 100)
        if value > self._threshold:
            value = self._threshold
        elif value < - self._threshold:
            value = - self._threshold
        return value + self._threshold


class Pipeline:
    def __init__(self):
        self.filters = [NaNFilter(),
                        ContinueUpFilter(high_idx=1, low_idx=2),
                        RatioNorm({'open': 0, 'high': 1, 'low': 2,
                                   'close': 3, 'volume': 4}, use_log1p=True)]

    def __call__(self, data):
        data = data.copy()
        for filter in self.filters:
            data = filter(data)
            if data is None:
                return None
        return data


if __name__ == '__main__':
    test_data = np.arange(25).reshape(5, 5) + 1
    print(test_data)
    pipe = Pipeline()
    print(pipe(test_data))
