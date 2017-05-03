from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.utils.data_utils import get_file
import logging


class Kevnet(Model):
  logger = logging.getLogger('pdk.kevnet')
  
  def __init__(self, generator):

    self.generator = generator
    self.logger.info("Assembling Model")
    self._input = Input(shape=generator.output)
    self.logger.info(self._input.shape)

    layer = Conv3D(8, 3, activation='relu', padding='same',  data_format='channels_first',
                          name='block1_conv1')(self._input)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 2, 2), strides=(1,2, 2),  data_format='channels_first', name='block1_pool')(layer)
    self.logger.info(layer.shape)
    
    layer = Conv3D(16, 3, activation='relu', padding='same',  data_format='channels_first',
                          name='block2_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 2, 2), strides=(1,2, 2),  data_format='channels_first',name='block2_pool')(layer)
    self.logger.info(layer.shape)
    layer = Conv3D(32, 3, activation='relu', padding='same',   data_format='channels_first',
                          name='block3_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 2, 2), strides=(1,2, 2),  data_format='channels_first', name='block3_pool')(layer)

    self.logger.info(layer.shape)
    layer = Conv3D(64, 3, activation='relu', padding='same',   data_format='channels_first',
                          name='block4_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 2, 2), strides=(1,2, 2),  data_format='channels_first', name='block4_pool')(layer)

    self.logger.info(layer.shape)
    layer = Conv3D(128, 3, activation='relu', padding='same',   data_format='channels_first',
                          name='block5_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((2, 4, 4), strides=(1,4, 4),  data_format='channels_first', name='block5_pool')(layer)

    self.logger.info(layer.shape)
    layer = Conv3D(256, 3, activation='relu', padding='same',   data_format='channels_first',
                          name='block6_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((2, 4, 4), strides=(1,4, 4),  data_format='channels_first', name='block6_pool')(layer)
    

    # Classification block
    self.logger.info(layer.shape)
    layer = Flatten(name='flatten')(layer)
    layer = Dense(512, activation='relu', name='fc1')(layer)
    layer = Dense(512, activation='relu', name='fc2')(layer)
    layer = Dense(generator.input, activation='softmax', name='predictions')(layer)


    super(Kevnet, self).__init__(self._input, layer)
    self.logger.info("Compiling Model")
    self.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    



