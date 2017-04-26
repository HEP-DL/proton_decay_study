from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.utils.data_utils import get_file
import logging


class VGG16(Model):
  logger = logging.getLogger('pdk.vgg16')
  
  def __init__(self, generator):

    self.generator = generator
    self.logger.info("Assembling Model")
    # The input shape is defined as 3 planes at 576x576 pixels
    # TODO: I think with the Theano backend, this might need to be reversed.

    if K.image_dim_ordering() != 'th':
        self.logger.error("Dimension Ordering Incorrect")

    self._input = Input(generator.output)
    #self.logger.debug("Input Shape: {}".format(self._input.output_shape))

    # Block 1
    layer = Convolution2D(64, 3, 3, activation='relu', border_mode='same', 
                          name='block1_conv1')(self._input)
    layer = Convolution2D(64, 3, 3, activation='relu', border_mode='same', 
                          name='block1_conv2')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(layer)

    # Block 2
    layer = Convolution2D(128, 3, 3, activation='relu', border_mode='same', 
                          name='block2_conv1')(layer)
    layer = Convolution2D(128, 3, 3, activation='relu', border_mode='same', 
                          name='block2_conv2')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(layer)

    # Block 3
    layer = Convolution2D(256, 3, 3, activation='relu', border_mode='same', 
                          name='block3_conv1')(layer)
    layer = Convolution2D(256, 3, 3, activation='relu', border_mode='same', 
                          name='block3_conv2')(layer)
    layer = Convolution2D(256, 3, 3, activation='relu', border_mode='same', 
                          name='block3_conv3')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(layer)

    # Block 4
    layer = Convolution2D(512, 3, 3, activation='relu', border_mode='same', 
                          name='block4_conv1')(layer)
    layer = Convolution2D(512, 3, 3, activation='relu', border_mode='same', 
                          name='block4_conv2')(layer)
    layer = Convolution2D(512, 3, 3, activation='relu', border_mode='same', 
                          name='block4_conv3')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(layer)

    # Block 5
    layer = Convolution2D(512, 3, 3, activation='relu', border_mode='same', 
                          name='block5_conv1')(layer)
    layer = Convolution2D(512, 3, 3, activation='relu', border_mode='same', 
                          name='block5_conv2')(layer)
    layer = Convolution2D(512, 3, 3, activation='relu', border_mode='same', 
                          name='block5_conv3')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(layer)

    # Classification block
    layer = Flatten(name='flatten')(layer)
    layer = Dense(4096, activation='relu', name='fc1')(layer)
    layer = Dense(4096, activation='relu', name='fc2')(layer)
    layer = Dense(generator.input, activation='softmax', name='predictions')(layer)
    
    super(VGG16, self).__init__(self._input, layer)
    self.logger.info("Compiling Model")
    self.compile(loss='binary_crossentropy', optimizer='sgd')
"""
  def train_with_incremental_save(self, samples_per_epoch, n_epochs_total, epoch_per_save):
    #loop over the trainings and save to file
    total_epoch



    training_output = model.fit_generator(self._input, samples_per_epoch = samples_per_epoch, 
                                      nb_epoch=n_epochs_total)
"""