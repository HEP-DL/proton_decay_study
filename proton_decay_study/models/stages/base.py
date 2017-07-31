from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.layers import Input, Dropout, Dense, Flatten
from keras.models import Model
from keras import optimizers
from keras.regularizers import l1, l2
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
  fc_name='fc_base'

  def __init__(self, generator):
    
    self.generator = generator
    layer = self.assemble()
    super(BaseNet, self).__init__(self._input, layer)
    self.sgd = optimizers.RMSprop(lr=1e-9, rho=0.999, epsilon=1e-9, decay=1e-9)
    self.logger.info("Compiling...")
    #could also do mean_squared_error
    self.compile(loss='kullback_leibler_divergence', optimizer=self.sgd,
                 metrics=['accuracy'])
    self.logger.info("Finished Compiling Network")

  def pre_assemble(self):
    self._input = Input(shape=self.generator.output,
                        dtype='float32',
                        name='input_1')
    self.logger.info(self._input)
    return self._input

  def post_assemble(self, layer):
    layer = Flatten(name='flatten_{}'.format(self.fc_name))(layer)
    self.logger.info(layer)
    layer = Dense(1024, self.generator.input,
                  activation='sigmoid',
                  kernel_initializer='random_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=l2(0.01),
                  activity_regularizer=l1(0.01),
                  name=self.fc_name)(layer)
    self.logger.info(layer)
    return layer

  def assemble_layers(self, layer):
    return layer

  def assemble(self):

    layer = self.pre_assemble()
    layer = self.assemble_layers(layer)
    layer = self.post_assemble(layer)
    return layer    

