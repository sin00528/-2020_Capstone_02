import keras
import tensorflow as tf

from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Lambda
from keras.layers import Input, Reshape, LeakyReLU, ZeroPadding2D, Activation, Add
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import categorical_crossentropy
from keras.models import Model, load_model, Sequential
from collections import deque

from tensorflow_addons.layers import InstanceNormalization

import sys
import numpy as np


def make_encoder(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(32, (7, 7), strides=(1, 1), padding='same',
        activation='relu', input_shape=input_shape))
    
    model.add(InstanceNormalization(axis=3, center=True, scale=True))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same',
        activation='relu'))
    model.add(InstanceNormalization(axis=3, center=True, scale=True))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same',
        activation='relu'))
    
    model.add(InstanceNormalization(axis=3, center=True, scale=True))

    return model

def make_decoder(input_shape=(32, 32, 128)):
    # Decoder/Generator
    inputs = Input(shape=input_shape)
    
    # 6 Residual blocks
    shortcut = inputs
    for i in range(6):
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(shortcut)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Add()([x, shortcut])     
        x = Activation('relu')(x)

        if i != 5:
            shortcut = x
    
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = InstanceNormalization()(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = InstanceNormalization()(x)

    outputs = Conv2D(3, (7, 7), strides=(1, 1), padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def make_discriminator(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    
    model.add(LeakyReLU())
    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(1024, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(2048, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())

    model.add(Conv2D(2048, (3, 3), strides=(3, 3), padding='same'))

    return model

def make_classifier(num_classes, input_shape=(32, 32, 128)):
    model = Sequential()
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same',
        activation='relu', input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(384, (5, 5), strides=(1, 1), padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))

    model.add(Dense(num_classes))

    max = 1
    min = -1
    model.add(Lambda(lambda x: (x - min) / (max - min)))

    return model