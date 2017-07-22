"""
    Defines the default callbacks for usage in the mod:`proton_decay_study`
"""
import json
from keras.callbacks import Callback
import time
import csv
import os
import six
from collections import OrderedDict


class HistoryRecord(Callback):

    def __init__(self, filename, n_iter=10,separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        self.epoch = 0
        self.time_begin = None
        self.n_iter = n_iter
        super(HistoryRecord, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)
        self.time_begin = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch=epoch

    def on_batch_end(self, batch, logs=None):
        if not batch%self.n_iter==0:
            return
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch','batch','time'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'batch': batch, 
                                'epoch': self.epoch,
                                'time': time.time()-self.time_begin})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
