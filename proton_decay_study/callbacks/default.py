import history
import json
from keras.callbacks import Callback
import keras

class HistoryRecord(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy =[]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))

    def write(self, filename):
      with open (filename, 'w') as output:
        output.write(json.dumps({'loss':self.losses, 'accuracy': self.accuracy}))
