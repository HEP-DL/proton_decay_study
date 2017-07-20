from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.layers import Input, Dropout, Dense, Flatten
from keras.models import Model
from keras import optimizers
import logging


class BaseNet(Model):
  logger = logging.getLogger('pdk.basenet')

  def __init__(self, generator):

    self.generator = generator

    layer = self.assemble()
    super(BaseNet, self).__init__(self._input, layer)
    self.logger.info("Compiling Model")
    self.sgd = optimizers.SGD(lr=0.1,
                              decay=1e-3,
                              momentum=0.5, 
                              nesterov=True)
    self.compile(loss='mean_squared_error', optimizer=self.sgd,
                 metrics=['accuracy'])

  def pre_assemble(self):
    self._input = Input(shape=self.generator.output,
                        dtype='float32')
    return self._input

  def post_assemble(self, layer):
    layer = Dense(self.generator.input,
                  activation='softmax')(layer)
    self.logger.info(layer)
    return layer

  def assemble_layers(self, layer):
    return layer

  def assemble(self):

    layer = self.pre_assemble()
    layer = self.assemble_layers(layer)
    layer = self.post_assemble(layer)
    return layer    

