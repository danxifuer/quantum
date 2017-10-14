
import tushare as ts
from data_process import _norm_max_min
import numpy as np
from rnn_config import *
import pickle

all_code = ts.get_stock_basics()
print(len(all_code.index.values))

all_val_data = []
for code in all_code.index.values:
    # open   high  close    low   volume    amount
    # ["open", "close", "high", "low", "total_turnover", "volume"]
    data = ts.get_h_data(code=code, start='2017-08-10', end='2017-10-14', retry_count=20)
    data = data[['open', 'close', 'high', 'low', 'amount', 'volume']]
    np_data = data.values
    # print(data[-1, 1])
    if np_data.shape[0] < PREDICT_LEN + 1:
        print('data miss')
        continue
    for i in range(0, np_data.shape[0] - PREDICT_LEN - 1, 1):
        sample = np_data[i: i + PREDICT_LEN]
        norm_data = _norm_max_min(sample, copy=True)
        norm_data = np.array([norm_data])
        up_ratio = np_data[i + PREDICT_LEN, 1] / np_data[i + PREDICT_LEN - 1, 1]
        real = int(up_ratio > 1)
        all_val_data.append((norm_data, real))

pickle.dump(all_val_data, open('./val_data.pkl', 'wb'))

