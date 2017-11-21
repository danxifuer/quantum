from matplotlib.pylab import date2num
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.finance as mpf
from photon.db_access import get_ochlv_for_k_line


def date_to_num(dates):
    num_time = []
    for date in dates:
        print(date)
        # date_time = datetime.datetime.strptime(date, '%Y-%m-%d')
        num_date = date2num(date)
        num_time.append(num_date)
    return np.array(num_time)

data = get_ochlv_for_k_line('sz000002', start_date='2017-10-16', end_date='2017-10-31')
data[:, 0] = date_to_num(data[:, 0])
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(15, 8))
mpf.candlestick_ochl(ax1, data, width=1.0, colorup='r', colordown='g')
ax1.set_title('code')
ax1.set_ylabel('Price')
ax1.grid(True)
ax1.xaxis_date()
plt.bar(data[:, 0] - 0.25, data[:, 5], width=0.5)
ax2.set_ylabel('Volume')
ax2.grid(True)
plt.show()
