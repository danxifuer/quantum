from database import ConnManage
import numpy as np

conn_manager = ConnManage()


def base_info(code):
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
    highest = np.max(close_p)
    lowest = np.min(close_p)
    from_date = trade_date_list[0]
    end_date = trade_date_list[-1]
    sql = conn_manager.exec_template('base_info', {'total_num': len(trade_date_list),
                                                   'code': '"%s"' % code,
                                                   'from_date': '"%s"' % from_date,
                                                   'end_date': '"%s"' % end_date,
                                                   'close_price_highest': highest,
                                                   'close_price_lowest': lowest,
                                                   })
    # print(sql)
    conn_manager.exec_sql(sql)


def _handle(code):
    code = code.split('.')[0]
    if code.startswith('6'):
        return 'sh%s' % code
    elif code.startswith('0') or code.startswith('3'):
        return 'sz%s' % code
    else:
        raise ValueError(code)


if __name__ == '__main__':
    code_list = open('all_code').readline().strip().split(',')
    code_list = map(_handle, code_list)
    count = 0
    for code in code_list:
        print(code)
        print(count)
        base_info(code)
        count += 1
