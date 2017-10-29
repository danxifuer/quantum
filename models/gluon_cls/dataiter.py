from queue import Queue
import threading
import mxnet as mx
import numpy as np
import time


class DataIter:
    def __init__(self, rec_file, batch_size):
        self.q = Queue()
        self.batch_size = batch_size
        self.pre_fetch_num = 10
        self.recordio = mx.recordio.MXRecordIO(rec_file, 'r')
        self.th = threading.Thread(target=self._load_data)
        self.th.daemon = True
        self.th.start()

    def _load_data(self):
        while True:
            if self.q.qsize() >= self.pre_fetch_num:
                time.sleep(0.01)
            batch_data = []
            batch_label = []
            for _ in range(self.batch_size):
                item = self.recordio.read()
                if item is None:
                    self.recordio.reset()
                    item = self.recordio.read()
                header, data = mx.recordio.unpack(item)
                array = np.frombuffer(data, np.float32).reshape(30, 7)
                batch_data.append(array)
                batch_label.append(header.label)
            batch_data = mx.nd.array(batch_data)
            batch_label = mx.nd.array(batch_label)
            self.q.put((batch_data, batch_label))

    def next(self):
        return self.q.get(block=True)

