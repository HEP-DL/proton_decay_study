from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.utils.data_utils import get_file
import logging


class InceptionV4(Model):
    logger = logging.getLogger('pdk.model.inceptionv4')

    def __init__(self, generator):
        self._input = Input(generator.output)
        layer = self.conv_block(self._input, 32, 3, 3, subsample = (2,2), border_mode = 'valid')
        layer = self.conv_block(layer, 32, 3, 3, border_mode='valid')
        layer = self.conv_block(layer, 64, 3, 3)

        layer_1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(layer)
        layer_2 = self.conv_block(layer, 96, 3, 3, subsample=(2, 2), border_mode='valid')

        layer = merge([layer_1, layer_2], mode='concat', concat_axis=-1)

        layer_1 = self.conv_block(layer, 64, 1, 1)
        layer_1 = self.conv_block(layer_1, 96, 3, 3, border_mode='valid')

        layer_2 = self.conv_block(layer, 64, 1, 1)
        layer_2 = self.conv_block(layer_2, 64, 1, 7)
        layer_2 = self.conv_block(layer_2, 64, 7, 1)
        layer_2 = self.conv_block(layer_2, 96, 3, 3, border_mode='valid')

        layer = merge([layer_1, layer_2], mode='concat', concat_axis=-1)

        layer_1 = self.conv_block(layer, 192, 3, 3, subsample=(2, 2), border_mode='valid')
        layer_2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(layer)

        layer = merge([layer_1, layer_2], mode='concat', concat_axis=-1)


        for i in range(4):

            layer_1 = self.conv_block(layer, 96, 1, 1)

            layer_2 = self.conv_block(layer, 64, 1, 1)
            layer_2 = self.conv_block(layer_2, 96, 3, 3)

            layer_3 = self.conv_block(layer, 64, 1, 1)
            layer_3 = self.conv_block(layer_3, 96, 3, 3)
            layer_3 = self.conv_block(layer_3, 96, 3, 3)

            layer_4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(layer)
            layer_4 = self.conv_block(layer_4, 96, 1, 1)

            layer = merge([layer_1, layer_2, layer_3, layer_4], mode='concat', concat_axis=-1)


        layer_1 = self.conv_block(layer, 384, 3, 3, subsample=(2, 2), border_mode='valid')

        layer_2 = self.conv_block(layer, 192, 1, 1)
        layer_2 = self.conv_block(layer_2, 224, 3, 3)
        layer_2 = self.conv_block(layer_2, 256, 3, 3, subsample=(2, 2), border_mode='valid')

        layer_3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(layer)

        layer = merge([layer_1, layer_2, layer_3], mode='concat', concat_axis=-1)


        for i in range(7):
            layer_1 = self.conv_block(layer, 384, 1, 1)

            layer_2 = self.conv_block(layer, 192, 1, 1)
            layer_2 = self.conv_block(layer_2, 224, 1, 7)
            layer_2 = self.conv_block(layer_2, 256, 7, 1)

            layer_3 = self.conv_block(layer, 192, 1, 1)
            layer_3 = self.conv_block(layer_3, 192, 7, 1)
            layer_3 = self.conv_block(layer_3, 224, 1, 7)
            layer_3 = self.conv_block(layer_3, 224, 7, 1)
            layer_3 = self.conv_block(layer_3, 256, 1, 7)

            layer_4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(layer)
            layer_4 = self.conv_block(layer_4, 128, 1, 1)

            layer = merge([layer_1, layer_2, layer_3, layer_4], mode='concat', concat_axis=-1) 

        layer_1 = self.conv_block(layer, 192, 1, 1)
        layer_1 = self.conv_block(layer_1, 192, 3, 3, subsample=(2, 2), border_mode='valid')

        layer_2 = self.conv_block(layer, 256, 1, 1)
        layer_2 = self.conv_block(layer_2, 256, 1, 7)
        layer_2 = self.conv_block(layer_2, 320, 7, 1)
        layer_2 = self.conv_block(layer_2, 320, 3, 3, subsample=(2, 2), border_mode='valid')

        layer_3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(layer)

        layer = merge([layer_1, layer_2, layer_3], mode='concat', concat_axis=-1)

    
        for i in range(3):
            layer_1 = conv_block(layer, 256, 1, 1)

            layer_2 = conv_block(layer, 384, 1, 1)
            layer_2_1 = conv_block(layer_2, 256, 1, 3)
            layer_2_2 = conv_block(layer_2, 256, 3, 1)
            layer_2 = merge([layer_2_1, layer_2_2], mode='concat', concat_axis=-1)

            layer_3 = conv_block(layer, 384, 1, 1)
            layer_3 = conv_block(layer_3, 448, 3, 1)
            layer_3 = conv_block(layer_3, 512, 1, 3)
            layer_3_1 = conv_block(layer_3, 256, 1, 3)
            layer_3_2 = conv_block(layer_3, 256, 3, 1)
            layer_3 = merge([layer_3_1, layer_3_2], mode='concat', concat_axis=-1)

            layer_4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(layer)
            layer_4 = conv_block(layer_4, 256, 1, 1)

            m = merge([layer_1, layer_2, layer_3, layer_4], mode='concat', concat_axis=-1)

        layer = AveragePooling2D((8, 8))(layer)

        layer = Dropout(0.8)(layer)
        layer = Flatten()(layer)

        # Output
        layer = Dense(output_dim=generator.input, activation='softmax')(layer)

        super(InceptionV4, self).__init__(self._input, layer)
        self.compile(loss='binary_crossentropy', optimizer='sgd')

    def conv_block(self, layer, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
        layer = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(layer)
        layer = BatchNormalization(axis=-1)(layer)
        layer = Activation('relu')(layer)
        return x

