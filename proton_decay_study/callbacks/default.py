"""
    Defines the default callbacks for usage in the mod:`proton_decay_study`
"""
import json
from keras.callbacks import Callback


class HistoryRecord(Callback):
    """
    This is a stub in place for working on recording the training history
    """
    filename = "intermediate.json"
    def __init__(self, filename=None):
        if filename is not None:
            self.filename = filename

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.epochs = []
        self.steps = []
        self.epoch = 0
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.steps.append(self.batch)
        self.epochs.append(self.epoch)
        self.batch += 1

    def on_epoch_end(self, epoch, logs={}):
        self.batch = 0
        self.epochs +=1
        self.write(self.filename)

    def write(self, filename):
      with open(filename, 'w') as output:
        output.write(json.dumps({'loss': self.losses,
                                 'accuracy': self.accuracy,
                                 'epochs': self.epochs,
                                 'steps': self.steps}))
