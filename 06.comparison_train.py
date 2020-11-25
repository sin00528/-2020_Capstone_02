import os
import time
import random
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import keras
import keras.backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Input, Reshape, LeakyReLU, ZeroPadding2D, Activation, Add
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.applications.resnet50 import ResNet50

from model import make_encoder, make_decoder, make_discriminator, make_classifier


# NOTE: load annotaion sample
LABEL_PATH = './annotations/'
IMG_PATH = './img_seq/train/'
os.makedirs('./plt/', exist_ok=True)
PLT_PATH = './plt/'

EPOCHS = 50
RND_SEED = 777
BATCH_SIZE = 256
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNEL = 3

ENC_IN_SHAPE = (128, 128, 3)
DEC_IN_SHAPE = (32, 32, 128)
DIS_IN_SHAPE = (128, 128, 3)
CLS_IN_SHAPE = (32, 32, 128)

random.seed(RND_SEED)
np.random.seed(RND_SEED)
tf.random.set_seed(RND_SEED)

# 1. load dataset
label_path = os.path.join(LABEL_PATH, 'img_label.csv')
label = pd.read_csv(label_path)
label = label[:1280]

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

train_datagen = ImageDataGenerator(preprocessing_function=prep_fn, validation_split=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_dataframe(label,
                                                x_col = 'file_path',
                                                y_col= ['valence', 'arousal'],
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                class_mode="multi_output",
                                                subset='training')


validataion_set = train_datagen.flow_from_dataframe(label,
                                                x_col = 'file_path',
                                                y_col= ['valence', 'arousal'],
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                class_mode="multi_output",
                                                subset='validation')


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = Flatten()(x)
predictions = Dense(2)(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Training
history = model.fit_generator(training_set, epochs=EPOCHS, steps_per_epoch=BATCH_SIZE) #, validation_data=(val_X, val_y))

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Training for ' +str(EPOCHS)+ ' epochs')
#plt.legend(['train_loss', 'val_loss'], loc='upper right')
#plt.savefig(os.path.join(PLT_PATH,'resnet.png'))