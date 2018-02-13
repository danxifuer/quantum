import pandas as pd
import plotly
from plotly.graph_objs import Data, Figure

plotly.tools.set_credentials_file(username='daiab',
                                  api_key='0XbDDTqKb2D4r1bQUH6x')


class Analysis:
    def __init__(self):
        pass

    @staticmethod
    def _plot(x, y, name='figure', remote=False):
        trace = []
        if isinstance(y, (list, tuple)):
            for i in y:
                t = {
                    "x": x,
                    "y": i,
                    "name": 'origin_close',
                    "type": "scatter"
                }
                trace.append(t)
        else:
            t = {
                "x": x,
                "y": y,
                "name": 'origin_close',
                "type": "scatter"
            }
            trace.append(t)
        data = Data(trace)
        layout = {"xaxis": {"tickangle": 30}}
        fig = Figure(data=data, layout=layout)
        if remote:
            url = plotly.plotly.plot(fig, filename='%s.html' % name)
            print('url: ', url)
        else:
            plotly.offline.plot(fig, filename='%s.html' % name)

    def plot(self):
        csv_file = '/home/daiab/machine_disk/work/deepblue/data/201701/dc/tick/20170103/jd1703_20170103.csv'
        data = pd.read_csv(csv_file, index_col=2, encoding='GB2312')
        data.index = pd.DatetimeIndex(data.index)
        data = data.iloc[:, [5, 6]]
        # data = pd.DataFrame()
        data = data.rolling(window=10).sum()
        self._plot(data.index, [data.iloc[:, 0], data.iloc[:, 1]])


if __name__ == '__main__':
    a = Analysis()
    a.plot()
