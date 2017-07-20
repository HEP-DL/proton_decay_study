from .stage6 import Stage6
from keras.layers.convolutional import MaxPooling3D, Conv3D
import logging

class Stage7(Stage6):
  logger = logging.getLogger('pdk.stage7')

  def assemble_layers(self, layer):
    layer = super(Stage7, self).assemble_layers(layer)
    layer = Conv3D(1024, (1, 2, 1), strides=(1, 2, 1),
                   activation='relu', padding='same',
                   data_format='channels_first')(layer)
    self.logger.info(layer)
    layer = MaxPooling3D((1, 2, 1), strides=(1, 2, 1),
                         data_format='channels_first')(layer)
    self.logger.info(layer)
    return layer
