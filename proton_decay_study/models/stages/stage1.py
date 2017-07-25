from .base import BaseNet
from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.regularizers import l1, l2
import logging

class Stage1(BaseNet):
  logger = logging.getLogger('pdk.stage1')
  fc_name='fc_stage1'

  def assemble_layers(self, layer):
    layer = Conv3D(32, (1, 5, 5), strides=(1, 4, 4),
                       activation='relu', padding='same',
                       data_format='channels_first',
                       kernel_initializer='random_uniform',
                       bias_initializer='zeros',
                       kernel_regularizer=l2(0.01),
                       activity_regularizer=l1(0.01),
                       name='conv1_1')(self._input)
    self.logger.info(layer)
    layer = MaxPooling3D((1, 4, 4), strides=(1, 4, 4),
                         data_format='channels_first',
                         name='pool_1_1')(layer)
    self.logger.info(layer)
    return layer
