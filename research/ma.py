import pandas as pd
from indicator.simple_moving_average import simple_moving_average as sma
import plotly
from plotly.figure_factory import create_candlestick
from plotly.graph_objs import Data, Figure
plotly.tools.set_credentials_file(username='daiab',
                                  api_key='0XbDDTqKb2D4r1bQUH6x')


def _plot_kline(data, remote=True):
    fig = create_candlestick(data.open, data.high, data.low, data.close, dates=data.index)
    if remote:
        url = plotly.plotly.plot(fig, filename='candle.html')
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename='candle.html', validate=False)


def _plot_ma(indexs, data_list, names, remote=True, filename='ma.html'):
    trace = []
    for i, d in enumerate(data_list):
        t = {
            "x": indexs[i],
            "y": d,
            "name": names[i],
            "type": "scatter"
        }
        trace.append(t)
    data = Data(trace)
    layout = {"xaxis": {"tickangle": 35}}
    fig = Figure(data=data, layout=layout)
    if remote:
        url = plotly.plotly.plot(fig, filename=filename)
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename=filename)


def candle_line(day_csv, remote=True):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    _plot_kline(day_rb, remote)


def between_days_ma(day_csv, remote=True):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    short = sma(day_rb['close'], 5)
    long = sma(day_rb['close'], 30)
    index = list(range(day_rb.shape[0]))
    _plot_ma((index, index),
             (short, long),
             ('day_short', 'day_long'),
             remote,
             'between_days_ma.html')


def between_day_min_ma(day_csv, min_csv, remote=True):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    min_rb = pd.read_csv(min_csv, index_col=0)
    min_rb.index = pd.DatetimeIndex(min_rb.index)
    short = sma(min_rb['close'], 5)
    long = sma(day_rb['close'], 2)
    # plot_kline(day_rb)
    _plot_ma((list(range(min_rb.shape[0])), list(range(day_rb.shape[0]))),
             (short, long),
             ('min_short', 'day_long'),
             remote,
             'between_day_min_ma.html')


def plot_origin_line(day_csv, remote=False):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    t = {
        "x": list(range(day_rb.shape[0])),
        "y": day_rb['close'].values,
        "name": 'origin_line',
        "type": "scatter"
    }
    data = Data([t])
    layout = {"xaxis": {"tickangle": 35}}
    fig = Figure(data=data, layout=layout)
    if remote:
        url = plotly.plotly.plot(fig, filename='ma.html')
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename='ma.html')


def skew_kurt(day_csv, remote=True):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    day_rb = day_rb['close']
    day_rb = (day_rb - day_rb.mean()) / day_rb.std()
    skew_value = day_rb.rolling(30).skew()
    kurt_value = day_rb.rolling(30).kurt()
    index = list(range(day_rb.shape[0]))
    t = {
        "x": index,
        "y": day_rb.values,
        "name": 'origin',
        "type": "scatter"
    }
    t2 = {
        "x": index,
        "y": skew_value.values,
        "name": 'skew',
        "type": "scatter"
    }
    t3 = {
        "x": index,
        "y": kurt_value.values,
        "name": 'kurt',
        "type": "scatter"
    }
    data = Data([t, t2, t3])
    layout = {"xaxis": {"tickangle": 30}}
    fig = Figure(data=data, layout=layout)
    if remote:
        url = plotly.plotly.plot(fig, filename='skew_kurt.html')
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename='skew_kurt.html')


if __name__ == '__main__':
    # between_day_min_ma('/home/daiab/machine_disk/code/quantum/database/RB_1day.csv',
    #                    '/home/daiab/machine_disk/code/quantum/database/RB_min.csv',
    #                    True)
    between_days_ma('/home/daiab/machine_disk/code/quantum/database/RB_1day.csv',
                    True)
    candle_line('/home/daiab/machine_disk/code/quantum/database/RB_1day.csv',
                True)
    skew_kurt('/home/daiab/machine_disk/code/quantum/database/RB_1day.csv')
    # plot_origin_line('/home/daiab/machine_disk/code/quantum/database/RB_1day.csv')

