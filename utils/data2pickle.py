import os
import pickle
import pandas


def dataframe2pickle(root_path, save_file, shift=0):
    files = os.listdir(root_path)
    all_data = []
    fields = ["open", "close", "high", "low", "volume"]
    for f in files:
        tmp = pandas.read_pickle(os.path.join(root_path, f))
        tmp = tmp[f[:f.rfind('.')]]
        # avoid the affect by list on trade market
        tmp = tmp[fields]['2010-01-01':]
        # print(tmp.values)
        # print(tmp)
        # exit()
        all_data.append(tmp.values)
    print(len(all_data))
    pickle.dump(all_data, open(save_file, 'wb'))


if __name__ == '__main__':
    root_path = '/home/daiab/machine_disk/data/tushare_data/1d'
    save_file = '/home/daiab/machine_disk/data/tushare_data/rq_1d_from_2010.pkl'
    dataframe2pickle(root_path, save_file, shift=10)



