from .stage2 import Stage2
from keras.layers.convolutional import MaxPooling3D, Conv3D
import logging

class Stage3(Stage2):
  logger = logging.getLogger('pdk.stage3')

  def assemble_layers(self, layer):
    layer = super(Stage3, self).assemble_layers(layer)
    layer = Conv3D(128, (1, 3, 3), strides=(1, 2, 2),
                   activation='relu', padding='same',
                   data_format='channels_first')(layer)
    self.logger.info(layer)
    layer = MaxPooling3D((1, 2, 2), strides=(1, 2, 2),
                         data_format='channels_first')(layer)
    self.logger.info(layer)
    return layer