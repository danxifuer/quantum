import pandas as pd
import os
from datetime import datetime, timedelta


def remove_ms_for_rb(csv_file):
    output_file = csv_file + '.tmp'
    if os.path.exists(output_file):
        return output_file
    fo = open(output_file, 'w')
    with open(csv_file) as fi:
        for l in fi:
            l = l.split(',')
            t = l[0]
            t = t[:t.rfind('.')]
            l[0] = t
            fo.write(','.join(l))
    fo.close()
    return output_file


def remove_from_pandas(csv_file, remove_num=1150):
    rb = pd.read_csv(csv_file, index_col=0)
    s_date = datetime(2014, 9, 9, 8, 59, 00)
    rb.index = pd.DatetimeIndex(rb.index)
    print('origin shape', rb.shape)
    drop_index = []
    for i in range(remove_num):
        delta = timedelta(i)
        e_date = s_date + delta
        timestamp = pd.Timestamp(e_date)
        drop_index.append(timestamp)

    rb = rb.drop(drop_index, errors='ignore')
    rb.to_csv(csv_file)
    print('after shape', rb.shape)
    return csv_file


def min_line(csv_file, save_file, min_time=5, end_date=datetime(2017, 9, 6)):
    remove_num = min_time
    rb = pd.read_csv(csv_file, index_col=0)
    rb.index = pd.DatetimeIndex(rb.index)
    rb_output = pd.DataFrame(columns=rb.columns)
    s_date = datetime(2014, 9, 9, 9, 1, 00)
    delta = timedelta(minutes=min_time - 1)
    one_min = timedelta(minutes=1)
    while s_date < end_date:
        e_date = s_date + delta
        data = rb[s_date: e_date]
        s_date = e_date + one_min
        if data.shape[0] < remove_num:
            continue
        high = data['high'].max()
        low = data['low'].min()
        close = data.iloc[-1, 3]
        open = data.iloc[0, 0]
        volume = data['volume'].sum()
        rb_output.loc[e_date] = (open, high, low, close, volume)
    rb_output.to_csv(save_file)


def day_line(csv_file, save_file, day=1, end_date=datetime(2017, 9, 6)):
    remove_num = 10
    rb = pd.read_csv(csv_file, index_col=0)
    rb.index = pd.DatetimeIndex(rb.index)
    rb_output = pd.DataFrame(columns=rb.columns)
    s_date = datetime(2014, 9, 9)
    delta = timedelta(days=day)
    one_sec = timedelta(seconds=1)
    while s_date < end_date:
        e_date = s_date + delta - one_sec
        data = rb[s_date: e_date]
        s_date = e_date + one_sec
        if data.shape[0] < remove_num:
            continue
        high = data['high'].max()
        low = data['low'].min()
        close = data.iloc[-1, 3]
        open = data.iloc[0, 0]
        volume = data['volume'].sum()
        rb_output.loc[e_date] = (open, high, low, close, volume)
    rb_output.to_csv(save_file)


def concat_day_min(day_csv, min_csv, pre_days, min_duration=4):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    min_rb = pd.read_csv(min_csv, index_col=0)
    min_rb.index = pd.DatetimeIndex(min_rb.index)
    day_date = day_rb.index
    one_sec = timedelta(seconds=1)
    X = []
    Y = []
    # print(day_date[:30])
    for date in day_date[pre_days:-1]:
        idx = day_rb.index.searchsorted(date)
        start_idx = idx - pre_days
        min_start_idx = idx - min_duration
        day_data = day_rb[start_idx: min_start_idx + 1]
        min_start_date = day_date[min_start_idx] + one_sec
        min_end_date = date + one_sec
        min_data = min_rb[min_start_date: min_end_date]
        day_data = (day_data - day_data.mean()) / day_data.std()
        min_data = (min_data - min_data.mean()) / min_data.std()
        total_data = pd.concat((day_data, min_data))
        X.append(total_data)
        Y.append(day_rb.iloc[idx+1, 3])
        # print(total_data.shape)
        # print(day_rb.iloc[idx+1, 3].shape)
    return X, Y


def main():
    csv = remove_ms_for_rb('/home/daiab/machine_disk/code/quantum/database/RB_min.csv')
    csv = remove_from_pandas(csv)
    # min_line(csv,
    #          '/home/daiab/machine_disk/code/quantum/database/RB_30min.csv',
    #          min_time=30)
    day_line(csv,
             '/home/daiab/machine_disk/code/quantum/database/RB_1day.csv')


if __name__ == '__main__':
    # main()
    concat_day_min('/home/daiab/machine_disk/code/quantum/database/RB_1day.csv',
                   '/home/daiab/machine_disk/code/quantum/database/RB_30min.csv',
                   20,
                   4)
