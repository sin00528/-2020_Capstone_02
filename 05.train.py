import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

from model import VAE_GAN
from loss import cls_loss, dispel_loss, pixel_loss, adv_loss

# NOTE: load annotaion sample
LABEL_PATH = './annotations/'
IMG_PATH = './img_seq/train/'
OUT_PATH = './gan_images/'

EPOCH = 100
RND_SEED = 42
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128

random.seed(RND_SEED)
np.random.seed(RND_SEED)
tf.random.set_seed(RND_SEED)

label_path = os.path.join(LABEL_PATH, 'img_label.csv')
label = pd.read_csv(label_path)

print(label)

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)

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

"""
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_dataframe(label,
                                            x_col = 'file_path',
                                            y_col= ['valence', 'arousal'],
                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                            batch_size=BATCH_SIZE,
                                            class_mode="multi_output")
"""

# encoders
img_exp_encoder = VAE_GAN(model='encoder')
img_id_encoder = VAE_GAN(model='encoder')

# decoder
img_decoder = VAE_GAN(model='decoder')

# discriminator
discriminator = VAE_GAN(model='discriminator')

# classifier
classifier = VAE_GAN(model='classifier')



