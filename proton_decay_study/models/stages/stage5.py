from .stage4 import Stage4
from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.regularizers import l1, l2
import logging

class Stage5(Stage4):
  logger = logging.getLogger('pdk.stage5')
  fc_name='fc_stage5'

  def assemble_layers(self, layer):
    layer = super(Stage5, self).assemble_layers(layer)
    layer = Conv3D(256, (1, 3, 3), strides=(1, 2, 2),
                   activation='relu', padding='same',
                   data_format='channels_first',
                   kernel_initializer='random_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2(0.01),
                   activity_regularizer=l1(0.01),
                   name='conv5_1')(layer)
    self.logger.info(layer)
    layer = MaxPooling3D((1, 2, 2), strides=(1, 2, 2),
                         data_format='channels_first',
                         name='pool5_1')(layer)
    self.logger.info(layer)
    return layer
