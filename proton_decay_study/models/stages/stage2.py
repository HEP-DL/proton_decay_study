from .stage1 import Stage1
from keras.layers.convolutional import MaxPooling3D, Conv3D
import logging

class Stage2(Stage1):
  logger = logging.getLogger('pdk.stage2')

  def assemble_layers(self, layer):
    layer = super(Stage2, self).assemble_layers(layer)
    layer = Conv3D(64, (1, 3, 3), strides=(1, 2, 2),
                   activation='relu', padding='same',
                   data_format='channels_first',
                   kernel_initializer='random_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2(0.01),
                   activity_regularizer=l1(0.01))(layer)
    self.logger.info(layer)
    layer = MaxPooling3D((1, 2, 2), strides=(1, 2, 2),
                         data_format='channels_first')(layer)
    self.logger.info(layer)
    return layer
