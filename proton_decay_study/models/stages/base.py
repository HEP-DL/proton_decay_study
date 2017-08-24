from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.layers import Input, Dropout, Dense, Flatten
from keras.models import Model
from keras import optimizers
from keras.regularizers import l1, l2
import logging


class BaseNet(Model):
  logger = logging.getLogger('pdk.basenet')
  fc_name = 'fc_base'

  def __init__(self, generator):
    self.generator = generator
    layer = self.assemble()
    super(BaseNet, self).__init__(self._input, layer)
    self.rms_prop = optimizers.RMSprop(lr=1e-9, rho=0.999, epsilon=1e-9, decay=1e-9)
    self.logger.info("Compiling...")
    self.compile(loss='kullback_leibler_divergence', optimizer=self.rms_prop,
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
    layer = Dense(self.generator.input,
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

