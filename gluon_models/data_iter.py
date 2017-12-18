from queue import Queue
import threading
import mxnet as mx
import numpy as np
import time


class RecDataIter:
    def __init__(self, rec_file, batch_size, seq_len, input_size):
        self._q = Queue()
        self._batch_size = batch_size
        self._pre_fetch_num = 100
        self._seq_len = seq_len
        self._input_size = input_size
        self._recordio = mx.recordio.MXRecordIO(rec_file, 'r')
        th = threading.Thread(target=self._load_data)
        th.daemon = True
        th.start()

    def _load_data(self):
        while True:
            if self._q.qsize() >= self._pre_fetch_num:
                time.sleep(0.01)
                continue
            batch_data = []
            batch_label = []
            for _ in range(self._batch_size):
                item = self._recordio.read()
                if item is None:
                    self._recordio.reset()
                    item = self._recordio.read()
                header, data = mx.recordio.unpack(item)
                array = np.frombuffer(data, np.float32).reshape(self._seq_len, self._input_size)
                batch_data.append(array)
                batch_label.append(header.label)
            batch_data = np.array(batch_data)
            batch_label = np.array(batch_label)
            self._q.put((batch_data, batch_label))

    def next(self):
        return self._q.get(block=True)


class NDArrayDataIter:
    def __init__(self, nd_array, batch_size, seq_len):
        self._nd_array = nd_array
        self._size = nd_array.shape[0]
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._idx = 0

    def next(self):
        start = self._idx % (self._size - self._batch_size)
        end = start + self._batch_size
        data = self._nd_array[start, end]
        self._idx += 0
        return data
