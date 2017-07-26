from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.utils.data_utils import get_file

import tensorflow as tf
import logging


class Nothinbutnet(Model):
  logger = logging.getLogger('pdk.Nothinbutnet')
  
  def __init__(self, generator):

    self.generator = generator
    self.logger.info("Assembling Model")
    self._input = Input(shape=generator.output)
    self.logger.info(self._input)
    self.logger.info(self._input.shape)

    layer = Conv3D(32, (3,3,1), strides=(1,2,1), 
                   activation='relu', padding='valid', #'same', 
                   data_format='channels_first',
                   name='block1_conv1')(self._input)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 3, 3), strides=(1,2, 2),
                          data_format='channels_first', 
                          name='block1_pool')(layer)
    self.logger.info(layer.shape)
    layer = BatchNormalization(axis=2, name="block1_norm")(layer)
    self.logger.info(layer.shape)
    layer = Dropout(0.1)(layer)

    layer = Conv3D(64, (1,3,3), strides=(1,2,2), 
                   activation='relu', padding='same', 
                   data_format='channels_first',
                   name='block2_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 3, 3), strides=(1,2, 2),  
                          data_format='channels_first', 
                          name='block2_pool')(layer)
    self.logger.info(layer.shape)

    layer = Conv3D(128, (1,5,5), strides=(1,2,2), 
                   activation='relu', padding='same', 
                   data_format='channels_first',
                   name='block3_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 8, 8), strides=(1,4, 4),  
                          data_format='channels_first', 
                          name='block3_pool')(layer)
    self.logger.info(layer.shape)
    layer = BatchNormalization(axis=2, name="block3_norm")(layer)
    self.logger.info(layer.shape)
    layer = Dropout(0.1)(layer)

    layer = Conv3D(256, (1,3,3), strides=(1,2,2), 
                   activation='relu', padding='same', 
                   data_format='channels_first',
                   name='block4_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 3, 3), strides=(1, 2, 2),  
                          data_format='channels_first', 
                          name='block4_pool')(layer)
    self.logger.info(layer.shape)

    layer = Conv3D(512, (1,3,3), strides=(1,2,2), 
                   activation='relu', padding='same', 
                   data_format='channels_first',
                   name='block5_conv1')(layer)
    self.logger.info(layer.shape)
    layer = MaxPooling3D((1, 3, 3), strides=(1,2, 2),  
                          data_format='channels_first', 
                          name='block5_pool')(layer)
    self.logger.info(layer.shape)
    layer = Dropout(0.1)(layer)

    # Classification block
    layer = Flatten(name='flatten')(layer)
    layer = Dense(1024, activation='relu', name='fc1')(layer)
    layer = Dense(generator.input, activation='softmax', name='predictions')(layer)
    self.logger.info(layer.shape)

    super(Nothinbutnet, self).__init__(self._input, layer)
    self.logger.info("Compiling Model")
    self.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])



