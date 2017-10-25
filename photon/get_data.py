from photon.db_access import *
from random import shuffle
import pickle


def write_ohlcv_to(use_days, remove_head_num=10):
    code_list = get_code(greater_days=200)
    all_data = []
    for code in code_list:
        tmp = get_ohlcv(code)
        if len(tmp) <= remove_head_num:
            continue
        all_data.append(tmp[remove_head_num:])
    idx_list = []
    for i, d in enumerate(all_data):
        for s in range(len(d) - use_days):
            idx_list.append((i, s, s + use_days, s))
    shuffle(idx_list)
    for idx in idx_list:
