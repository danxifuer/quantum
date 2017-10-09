import os
import pickle
import pandas


ROOT_PATH = '/home/daiab/machine_disk/data/tushare_data/1d'

files = os.listdir(ROOT_PATH)

all_data = []
for f in files:
    tmp = pandas.read_pickle(os.path.join(ROOT_PATH, f))
    tmp = tmp[f[:f.rfind('.')]]
    # avoid the affect by list on trade market
    tmp = tmp.ix[10:]
    print(tmp)
    exit()
    all_data.append(tmp)

print(len(all_data))
pickle.dump(all_data, open('/home/daiab/machine_disk/data/tushare_data/rq_1d.pkl', 'wb'))


