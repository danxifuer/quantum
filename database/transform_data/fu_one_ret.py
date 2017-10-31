import numpy as np
from database import get_all_code, ConnManage

DAYS = 1

# TABLE = 'get_price'
TABLE = 'norm_data_across_stock'


conn_manager = ConnManage()


def cal_return(code, days):
    sql = "select close, trade_date from get_price where code = '%s' order by trade_date asc" % code
    query_result = conn_manager.query_sql(sql)
    if len(query_result) == 0:
        print('code: %s select result is null' % code)
        return
    close_p = []
    trade_date_list = []
    for items in query_result:
        close_p.append(items[0])
        trade_date_list.append(items[1])
    close_p = np.array(close_p, dtype=np.float32)
    trade_date_list = trade_date_list[:-days]
    returns = close_p[days:] / close_p[:-days]
    for i, trade_date in enumerate(trade_date_list):
        ret = returns[i]
        sql = "update {} set fu_one_ret={:.10f} " \
              " where code = '{}' and trade_date = '{}'".format(TABLE,
                ret, code, trade_date)
        conn_manager.exec_sql(sql)


if __name__ == '__main__':
    all_code = get_all_code(conn_manager)
    count = 0
    for code in all_code:
        print(code)
        print(count)
        cal_return(code, DAYS)
        count += 1
