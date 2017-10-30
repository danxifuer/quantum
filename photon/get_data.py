from photon.db_access import get_ohlcv_future_ret, get_code, get_normed_ohlcv_future_ret
from random import shuffle
import logging


def get_ohlcvr_and_shuffle_idx(use_days,
                               from_date='2008-01-01',
                               end_date='2017-07-01',
                               remove_head_num=0,
                               test_write=False):
    code_list = get_code(greater_days=200)
    all_data = []
    if test_write:
        logging.info('test_write just get some code')
        code_list = code_list[:3]
    query_count = 0
    for code in code_list:
        tmp = get_ohlcv_future_ret(code, from_date, end_date)
        if tmp.shape[0] <= (remove_head_num + use_days + 5):  # +5 is avoid error
            continue
        query_count += 1
        if query_count % 100 == 0:
            logging.info('query database: %s', query_count)
        all_data.append(tmp[remove_head_num:])
    idx_list = []
    for i, d in enumerate(all_data):
        for s in range(len(d) - use_days):
            idx_list.append((i, s, s + use_days))  # i th stock, start, end,
    shuffle(idx_list)
    return all_data, idx_list


def get_normed_ohlcvr_and_shuffled_idx(use_days,
                                       from_date='2008-01-01',
                                       end_date='2017-07-01',
                                       remove_head_num=0,
                                       test_write=False):
    code_list = get_code(greater_days=200)
    all_data = []
    if test_write:
        logging.info('test_write just get some code')
        code_list = code_list[:3]
    query_count = 0
    for code in code_list:
        tmp = get_normed_ohlcv_future_ret(code, from_date, end_date)
        if tmp.shape[0] <= (remove_head_num + use_days + 5):  # +5 is avoid error
            continue
        query_count += 1
        if query_count % 100 == 0:
            logging.info('query database: %s', query_count)
        all_data.append(tmp[remove_head_num:])
    idx_list = []
    for i, d in enumerate(all_data):
        for s in range(len(d) - use_days):
            idx_list.append((i, s, s + use_days))  # i th stock, start, end,
    shuffle(idx_list)
    return all_data, idx_list
