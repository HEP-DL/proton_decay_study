from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.layers import Input, Dropout, Dense, Flatten
from keras.models import Model
from keras import optimizers
import logging


class BaseNet(Model):
  """
  I said to the man on the door of my calm let me in
  It's time to begin
  He said, I must go to a town in the midst of it all
  Acoustics enthrall
  And now I have come to a place where my frequency's sold
  In Soundwaves of Gold
  """
  logger = logging.getLogger('pdk.basenet')

  def __init__(self, generator):

    self.generator = generator

    layer = self.assemble()
    super(BaseNet, self).__init__(self._input, layer)
    self.sgd = optimizers.SGD(lr=0.001,
                              decay=1e-10,
                              momentum=0.1, 
                              nesterov=True)
    # The other option here is mean square error
    self.compile(loss='mean_squared_error', optimizer=self.sgd,
                 metrics=['accuracy'])

  def pre_assemble(self):
    self._input = Input(shape=self.generator.output,
                        dtype='float32')
    self.logger.info(self._input)
    return self._input

  def post_assemble(self, layer):
    layer = Flatten(name='flatten')(layer)
    self.logger.info(layer)
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

