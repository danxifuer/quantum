from database import ConnManage
import numpy as np

conn_manager = ConnManage()


def norm_data():
    # sql = 'select code from base_info'
    ref_date = 'select trade_date from get_price where code = "sz000001"'
    all_trade_date = conn_manager.query_sql(ref_date)
    sql_template = 'select code, open, high, low, close, volume ' \
                   ' from get_price ' \
                   ' where trade_date = "%s" '
    sql_write_template = 'insert into norm_data_across_stock ' \
                         ' (h_o, l_o, c_o, o_c, h_c, l_c, volume, code, trade_date) ' \
                         ' values ({:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, "{}", "{}")'
    for date in all_trade_date:
        print('current date: %s' % date)
        sql = sql_template % date
        result = conn_manager.query_sql(sql)
        result = np.array(result)
        code = result[:, 0].astype(str)
        data = result[:, 1:].astype(np.float32)
        ret = np.empty(shape=(data.shape[0], 7))
        ret[:, 0] = np.log(data[:, 1] / data[:, 0])
        ret[:, 1] = np.log(data[:, 2] / data[:, 0])
        ret[:, 2] = np.log(data[:, 3] / data[:, 0])
        ret[:, 3] = np.log(data[:, 0] / data[:, 3])
        ret[:, 4] = np.log(data[:, 1] / data[:, 3])
        ret[:, 5] = np.log(data[:, 2] / data[:, 3])
        ret[:, 6] = data[:, 4]
        mean = np.mean(ret, axis=0, keepdims=True)
        std = np.std(ret, axis=0, keepdims=True)
        ret = (ret - mean) / std
        for i, c in enumerate(code):
            d = ret[i]
            sql = sql_write_template.format(d[0], d[1], d[2], d[3],
                                            d[4], d[5], d[6],
                                            c, '%s' % date)
            conn_manager.exec_sql(sql)


if __name__ == '__main__':
    norm_data()

