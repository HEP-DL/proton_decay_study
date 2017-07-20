from .base import BaseNet
from keras.layers.convolutional import MaxPooling3D, Conv3D
import logging

class Stage1(BaseNet):
  logger = logging.getLogger('pdk.stage1')

  def assemble_layers(self, layer):
    layer = Conv3D(64, (1, 5, 5), strides=(1, 5, 5),
                       activation='relu', padding='same',
                       data_format='channels_first')(self._input)
    self.logger.info(layer)
    layer = MaxPooling3D((1, 2, 2), strides=(1, 2, 2),
                         data_format='channels_first')(layer)
    self.logger.info(layer)
    return layer
