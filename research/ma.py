import pandas as pd
from indicator.simple_moving_average import simple_moving_average as sma
import plotly
from plotly.figure_factory import create_candlestick
from plotly.graph_objs import Data, Figure


def plot_kline(data):
    fig = create_candlestick(data.open, data.high, data.low, data.close, dates=data.index)
    plotly.offline.plot(fig, filename='candle', validate=False)


def plot_ma(index, data_list, names):
    trace = []
    for i, d in enumerate(data_list):
        t = {
            "x": index,
            "y": d,
            "name": names[i],
            "type": "scatter"
        }
        trace.append(t)
    data = Data(trace)
    layout = {"xaxis": {"tickangle": 35}}
    fig = Figure(data=data, layout=layout)
    plotly.offline.plot(fig)


def concat_day_min(day_csv, min_csv=None):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    if min_csv:
        min_rb = pd.read_csv(min_csv, index_col=0)
        min_rb.index = pd.DatetimeIndex(min_rb.index)
    ma_5day = sma(day_rb['close'], 5)
    ma_20day = sma(day_rb['close'], 20)
    # plot_kline(day_rb)
    plot_ma(day_rb.index, (ma_5day, ma_20day), ('5day', '20day'))
    # print(ma_5day)
    # print(ma_20day)


if __name__ == '__main__':
    concat_day_min('D:\quantum\database\RB_1day.csv')
