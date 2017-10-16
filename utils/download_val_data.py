from rqalpha.data.base_data_source import BaseDataSource
from rqalpha.data.data_proxy import DataProxy
from data_process import _norm_max_min
import datetime
import numpy as np
import tushare as ts
import pickle

PREDICT_LEN = 25
END_DATE = datetime.date(2017, 10, 10)


def get_data_from_rq(code, end_date, bar_count=20):
    data_source = BaseDataSource('/home/daiab/.rqalpha/bundle')
    data_proxy = DataProxy(data_source)
    # fields = ["open", "close", "high", "low", "total_turnover", "volume"]
    fields = ["open", "close", "high", "low", "volume"]
    data = data_proxy.history_bars(code,
                                   bar_count=bar_count,
                                   frequency='1d',
                                   field=fields,
                                   dt=end_date)
    print(data)
    return np.array(data.tolist())


all_code = ts.get_stock_basics()
all_val_data = []
for code in all_code.index.values:
    np_data = get_data_from_rq(code, end_date=END_DATE, bar_count=40)
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
