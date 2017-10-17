from rqalpha.data.base_data_source import BaseDataSource
from rqalpha.data.data_proxy import DataProxy
# from data_process import _norm_max_min
import datetime
import numpy as np
import pickle

END_DATE = datetime.date(2017, 10, 15)


class RQData:
    def __init__(self, end_date, bar_count=20):
        data_source = BaseDataSource('/home/daiab/.rqalpha/bundle')
        self._data_proxy = DataProxy(data_source)
        self.fields = ["open", "close", "high", "low", "volume"]
        self.end_date = end_date
        self.bar_count = bar_count

    def get_data(self, code):
        data = self._data_proxy.history_bars(code,
                                             bar_count=self.bar_count,
                                             frequency='1d',
                                             field=self.fields,
                                             dt=self.end_date)
        return np.array(data.tolist())


fi = open('/home/daiab/machine_disk/code/quantum/utils/all_code')
all_code = fi.readline().strip().split(',')
print('all_code length == %s' % len(all_code))
all_val_data = []
rqdata = RQData(END_DATE, bar_count=40)
for i, code in enumerate(all_code):
    np_data = rqdata.get_data(code)
    all_val_data.append(np_data)
    if i % 20 == 0:
        print(i)
pickle.dump(all_val_data, file=open('./val_data.pkl', 'wb'))
print('all val data == %s ' % len(all_val_data))
