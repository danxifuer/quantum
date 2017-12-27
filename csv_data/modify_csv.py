import pandas as pd

csv_file = '/home/daiab/Downloads/RB_min.csv'
rb = pd.read_csv(csv_file, index_col=0)
rb.index = pd.DatetimeIndex(rb.index)
idx_sort = ['open', 'high', 'low', 'close', 'volume']
rb = rb[idx_sort]
rb = rb[262000:]
# rb = rb[240000:262000]

print(rb.head(1))
print(rb.tail(1))
rb.to_csv('RB_min_infer.csv', sep=',', index=True)


# rb = pd.read_csv('RB_min_infer.csv', index_col=0)
# print(rb)
