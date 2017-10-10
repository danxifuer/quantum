import pickle
import numpy as np
THRESHOLD = 0.34

origin_data = pickle.load(open('/home/daiab/machine_disk/data/tushare_data/rq_1d.pkl', 'rb'))
origin_data = [df.values[:, 1].ravel() for df in origin_data if df is not None]
neg_num = 0
pos_num = 0
total = 0
for t in np.arange(1.0, 1.03, 0.001):
    for d in origin_data:
        ratio = d[1:] / d[:-1]
        pos = np.where(ratio >= t, 1, 0)
        pos_num += np.sum(pos)
        total += pos.shape[0]
    print('threshold: %s, ratio: %s' %(t, pos_num / total))
    pos_num = 0
    total = 0

print('=============================')
for t in np.arange(0.97, 1.0, 0.001):
    for d in origin_data:
        ratio = d[1:] / d[:-1]
        pos = np.where(ratio <= t, 1, 0)
        pos_num += np.sum(pos)
        total += pos.shape[0]
    print('threshold: %s, ratio: %s' %(t, pos_num / total))
    pos_num = 0
    total = 0
