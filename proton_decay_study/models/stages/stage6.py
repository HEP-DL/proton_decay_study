from .stage5 import Stage5
from keras.layers.convolutional import MaxPooling3D, Conv3D
import logging

class Stage6(Stage5):
  logger = logging.getLogger('pdk.stage6')

  def assemble_layers(self, layer):
    layer = super(Stage6, self).assemble_layers(layer)
    layer = Conv3D(512, (1, 3, 3), strides=(1, 2, 2),
                   activation='relu', padding='same',
                   data_format='channels_first')(layer)
    self.logger.info(layer)
    """
    layer = MaxPooling3D((1, 2, 2), strides=(1, 2, 2),
                         data_format='channels_first')(layer)
    self.logger.info(layer)
    """
    return layer
