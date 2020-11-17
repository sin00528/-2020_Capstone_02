import keras
import tensorflow as tf

from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Input, Reshape, LeakyReLU, ZeroPadding2D, Activation, Add
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, Conv2DTranspose
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.wrappers import TimeDistributed
from keras.losses import categorical_crossentropy
from keras.models import Model, load_model, Sequential
from collections import deque

from tensorflow_addons.layers import InstanceNormalization

import sys
import numpy as np
import keras.backend as K

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length, saved_model=None, features_length=2622):
        
        """
        @nb_classes: the number of classes to predict
        @seq_length: the length of our video sequences
        @saved_model: the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics.
        metrics=['mse', 'mae']

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'encoder':
            print("Loading encoder model.")
            #self.input_shape = (seq_length, 128, 128, 3)
            self.input_shape = (128, 128, 3)
            self.model = self.encoder()
        elif model == 'decoder':
            print("Loading decoder model.")
            #self.input_shape = (seq_length, 32, 32, 128)
            self.input_shape = (32, 32, 128)
            self.model = self.decoder()
        elif model == 'discriminator':
            print("Loading discriminator model.")
            #self.input_shape = (seq_length, 128, 128, 3)
            self.input_shape = (128, 128, 3)
            self.model = self.discriminator()
        elif model == 'classifier':
            print("Loading classifier model.")
            #self.input_shape = (seq_length, 128, 128, 3)
            self.input_shape = (128, 128, 3)
            self.model = self.classifier()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        #optimizer = RMSprop(lr=1e-3, decay=1e-6)
        optimizer = Adam(lr=1e-3, decay=1e-6)
        #optimizer = SGD(lr=1e-3) #, decay=1e-6)

        self.model.compile(loss='mse', optimizer=optimizer, metrics=metrics)
        print(self.model.summary())

        # NOTE: Need to check models padding

    # models
    def encoder(self):
        model = Sequential()
        model.add(Conv2D(32, (7, 7), strides=(1, 1), padding='same',
            activation='relu', input_shape=self.input_shape))
        
        model.add(InstanceNormalization())
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same',
            activation='relu'))
        model.add(InstanceNormalization())
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same',
            activation='relu'))
        
        model.add(InstanceNormalization())

        return model

    def decoder(self):
        inputs = Input(shape=self.input_shape)
        
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

            shortcut = x
        
        x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = InstanceNormalization()(x)
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = InstanceNormalization()(x)

        outputs = Conv2D(3, (7, 7), strides=(1, 1), padding='same')(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=self.input_shape))
        
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

    def classifier(self):
        model = Sequential()
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same',
            activation='relu', input_shape=self.input_shape))

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

        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

# NOTE: check summary
model = 'classifier'
rm = ResearchModels(2, model, seq_length=1, saved_model=None)