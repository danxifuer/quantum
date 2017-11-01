import requests
import json
import datetime
from database import ConnManage
import time

'''
https://gupiao.baidu.com/api/stocks/stockdaybar?
from=pc&os_ver=1&cuid=xxx&vv=100
&format=json&stock_code=sz000517&step=3
&start=&count=640&fq_type=front&timestamp=1509434605477

https://gupiao.baidu.com/api/stocks/stockdaybar?
from=pc&os_ver=1&cuid=xxx&vv=100
&format=json&stock_code=sz000012&step=3
&start=&count=160&fq_type=no&timestamp=1509503938702
'''
TIMESTAMP = 1509436066


def parse_item(item, code):
    time = item['date']
    kline = item['kline']
    open = kline['open']
    high = kline['high']
    low = kline['low']
    close = kline['close']
    volume = kline['volume']
    ratio = kline['netChangeRatio'] / 100 + 1
    trade_date = datetime.datetime.strptime(str(time), '%Y%m%d')
    return open, high, low, close, volume, code, trade_date


def get_code_data(code, timestamp, retry=10):
    count = 0
    while count < retry:
        try:
            json_data = requests.get('https://gupiao.baidu.com/api/stocks/stockdaybar',
                                     params={'from': 'pc',
                                             'os_ver': 1,
                                             'cuid': 'xxx',
                                             'format': 'json',
                                             'vv': 100,
                                             'stock_code': code,
                                             'step': 3,
                                             'start': '',
                                             'count': 4000,
                                             'fq_type': 'no',
                                             'timestamp': timestamp
                                             })
            ret = []
            json_data = json_data.content.decode()
            data = json.loads(json_data)
            if data['errorMsg'] == 'SUCCESS':
                items = data['mashData']
                for item in items:
                    ret.append(parse_item(item, code))
            return ret
        except:
            print('code error: %s' % code)
            time.sleep(0.1)
            count += 1


def _handle(code):
    code = code.split('.')[0]
    if code.startswith('6'):
        return 'sh%s' % code
    elif code.startswith('0') or code.startswith('3'):
        return 'sz%s' % code
    else:
        raise ValueError(code)


def write():
    conn_manager = ConnManage()
    code_list = open('all_code').readline().strip().split(',')
    # code_list = ['sz000517']sh
    code_list = map(_handle, code_list)
    sql_template = "insert into get_price " \
                   " (open, high, low, close, volume, code, trade_date)" \
                   " values (%s, %s, %s, %s, %s, '%s', '%s')"
    for i, code in enumerate(code_list):
        print('%s # %s' % (i, code))
        d = get_code_data(code, timestamp=TIMESTAMP)
        if d is None:
            print('code get no data: %s' % code)
            continue
        for item in d:
            sql = sql_template % item
            conn_manager.exec_sql(sql)


if __name__ == '__main__':
    write()
