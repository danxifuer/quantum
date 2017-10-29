def get_all_code(conn_manager):
    sql = 'select code from base_info'
    code = conn_manager.query_sql(sql)
    return [c[0] for c in code]

