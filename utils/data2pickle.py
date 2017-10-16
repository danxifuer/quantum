import os
import pickle
import pandas


def dataframe2pickle(root_path, save_file, shift=0):
    files = os.listdir(root_path)
    all_data = []
    for f in files:
        tmp = pandas.read_pickle(os.path.join(root_path, f))
        tmp = tmp[f[:f.rfind('.')]]
        # avoid the affect by list on trade market
        tmp = tmp.ix[shift:]
        print(tmp)
        exit()
        all_data.append(tmp)
    print(len(all_data))
    pickle.dump(all_data, open(save_file, 'wb'))



if __name__ == '__main__':
    root_path = '/home/daiab/machine_disk/data/tushare_data/1d'
    save_file = '/home/daiab/machine_disk/data/tushare_data/rq_1d.pkl'
    dataframe2pickle(root_path, save_file, shift=10)


import pandas as pd
import numpy as np

result = pd.DataFrame(np.arange(10).reshape(2, 5))
result.reset_index()
result.pivot_table()
pd.pivot_table()

