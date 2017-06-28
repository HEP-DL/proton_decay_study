"""
    Defines the default callbacks for usage in the mod:`proton_decay_study`
"""
import json
from keras.callbacks import Callback


class HistoryRecord(Callback):
    """
    This is a stub in place for working on recording the training history
    """
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))

    def write(self, filename):
      with open(filename, 'w') as output:
        output.write(json.dumps({'loss': self.losses,
                                 'accuracy': self.accuracy}))
