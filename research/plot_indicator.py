import pandas as pd
from indicator.double_exponential_moving_average import double_exponential_moving_average
from indicator.exponential_moving_average import exponential_moving_average
from indicator.moving_average_convergence_divergence import moving_average_convergence_divergence
from indicator.relative_strength_index import relative_strength_index
from indicator.simple_moving_average import simple_moving_average
from indicator.triangular_moving_average import triangular_moving_average
from indicator.triple_exponential_moving_average import triple_exponential_moving_average
from indicator.weighted_moving_average import weighted_moving_average
import plotly
from plotly.graph_objs import Data, Figure
from indicator.bollinger_bands import upper_bollinger_band, \
    lower_bollinger_band, middle_bollinger_band
from indicator.directional_indicators import average_directional_index
from indicator.aroon import aroon_up, aroon_down
from indicator.commodity_channel_index import commodity_channel_index
# from indicator.money_flow_index import money_flow_index
from indicator.momentum import momentum
from indicator.rate_of_change import rate_of_change
# from indicator.stochastic import percent_d, percent_k
from indicator.accumulation_distribution import accumulation_distribution
# from indicator.on_balance_volume import on_balance_volume
from indicator.standard_deviation import standard_deviation
from indicator.standard_variance import standard_variance
from indicator.average_true_range import average_true_range
from indicator.true_range import true_range

plotly.tools.set_credentials_file(username='daiab',
                                  api_key='0XbDDTqKb2D4r1bQUH6x')

DATA_PERIOD = {
    'DEMA': double_exponential_moving_average,
    'EMA': exponential_moving_average,
    'RSI': relative_strength_index,
    'SMA': simple_moving_average,
    'TRIMA': triangular_moving_average,
    'TEMA': triple_exponential_moving_average,
    'WMA': weighted_moving_average,
    'MOM': momentum,
    'ROC': rate_of_change,
    'ATR': average_true_range,
    'TRANGE': true_range,
    'VAR': standard_variance,
    'STD': standard_deviation
}

DATA_LONG_SHORT_PERIOD = {
    'MACD': moving_average_convergence_divergence
}

DATA_HIGHT_LOW_CLOSE = {
    'CCI': commodity_channel_index,
    'ADX': average_directional_index,
    'A/D Line': accumulation_distribution,
}


def plot_data_period(day_csv, index_name, max_period=30, remote=False):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    day_rb = day_rb['close']
    func = DATA_PERIOD[index_name]
    index = list(range(day_rb.shape[0]))
    t = {
        "x": index,
        "y": day_rb,
        "name": 'origin_close',
        "type": "scatter"
    }
    trace = [t]
    for i in range(2, max_period, 2):
        result = func(day_rb.values, period=i)
        t = {
            "x": index,
            "y": result,
            "name": '%s_%s_days' % (index_name, i),
            "type": "scatter"
        }
        trace.append(t)
    data = Data(trace)
    layout = {"xaxis": {"tickangle": 30}}
    fig = Figure(data=data, layout=layout)
    if remote:
        url = plotly.plotly.plot(fig, filename='%s.html' % index_name)
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename='%s.html' % index_name)


def plot_macd(day_csv, short_period, long_period, remote=False):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    day_rb = day_rb['close']
    index = list(range(day_rb.shape[0]))
    macd = moving_average_convergence_divergence(day_rb.values,
                                                 short_period,
                                                 long_period)
    t = {
        "x": index,
        "y": day_rb,
        "name": 'origin_close',
        "type": "scatter"
    }
    t1 = {
        "x": index,
        "y": macd,
        "name": '%s_%s_%s_days' % ('MACD', long_period, short_period),
        "type": "scatter"
    }
    data = Data([t, t1])
    layout = {"xaxis": {"tickangle": 30}}
    fig = Figure(data=data, layout=layout)
    if remote:
        url = plotly.plotly.plot(fig, filename='MACD.html')
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename='MACD.html')


def plot_close_high_low_period(day_csv, index_name, max_period=20, remote=False):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    close = day_rb['close']
    func = DATA_HIGHT_LOW_CLOSE[index_name]
    index = list(range(day_rb.shape[0]))
    t = {
        "x": index,
        "y": close,
        "name": 'origin_close',
        "type": "scatter"
    }
    trace = [t]
    for i in range(2, max_period, 2):
        result = func(day_rb['close'], day_rb['high'],
                      day_rb['low'], period=i)
        t = {
            "x": index,
            "y": result,
            "name": '%s_%s_days' % (index_name, i),
            "type": "scatter"
        }
        trace.append(t)
    data = Data(trace)
    layout = {"xaxis": {"tickangle": 30}}
    fig = Figure(data=data, layout=layout)
    if remote:
        url = plotly.plotly.plot(fig, filename='%s.html' % index_name)
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename='%s.html' % index_name)


def plot_bollinger_bands(day_csv, std_mult=2.0, period=20, remote=False):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    close = day_rb['close']
    upper = upper_bollinger_band(close.values, period, std_mult)
    middle = middle_bollinger_band(close.values, period, std_mult)
    lower = lower_bollinger_band(close.values, period, std_mult)
    index = list(range(day_rb.shape[0]))
    close = {
        "x": index,
        "y": close,
        "name": 'origin_close',
        "type": "scatter"
    }
    upper = {
        "x": index,
        "y": upper,
        "name": 'upper',
        "type": "scatter"
    }
    middle = {
        "x": index,
        "y": middle,
        "name": 'middle',
        "type": "scatter"
    }
    lower = {
        "x": index,
        "y": lower,
        "name": 'lower',
        "type": "scatter"
    }
    data = Data([close, upper, middle, lower])
    layout = {"xaxis": {"tickangle": 30}}
    fig = Figure(data=data, layout=layout)
    if remote:
        url = plotly.plotly.plot(fig, filename='bollinger_bands.html')
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename='bollinger_bands.html')


def plot_aroon(day_csv, period=20, remote=False):
    day_rb = pd.read_csv(day_csv, index_col=0)
    day_rb.index = pd.DatetimeIndex(day_rb.index)
    close = day_rb['close']
    up = aroon_up(close.values, period)
    down = aroon_down(close.values, period)
    index = list(range(day_rb.shape[0]))
    up = {
        "x": index,
        "y": up,
        "name": 'up',
        "type": "scatter"
    }
    down = {
        "x": index,
        "y": down,
        "name": 'down',
        "type": "scatter"
    }
    data = Data([up, down])
    layout = {"xaxis": {"tickangle": 30}}
    fig = Figure(data=data, layout=layout)
    if remote:
        url = plotly.plotly.plot(fig, filename='aroon.html')
        print('url: ', url)
    else:
        plotly.offline.plot(fig, filename='aroon.html')


if __name__ == '__main__':
    DAY_DATA = 'D:\quantum\csv_data\RB_1day.csv'
    # plot_data_period(DAY_DATA, "EMA")
    # plot_aroon(DAY_DATA, 20)
    plot_bollinger_bands(DAY_DATA, 2.0, 20)
    plot_macd(DAY_DATA, 2, 20)
    # plot_macd(DAY_DATA, 2, 10)
    # plot_close_high_low_period(DAY_DATA, 'A/D Line', 20)
