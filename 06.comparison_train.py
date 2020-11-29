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
from keras.layers import Dense, Flatten, Lambda
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.applications.resnet50 import ResNet50
from keras.applications import VGG16, VGG19

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

MDL_IN_SHAPE = (128, 128, 3)

random.seed(RND_SEED)
np.random.seed(RND_SEED)
tf.random.set_seed(RND_SEED)

# 1. load dataset
label_path = os.path.join(LABEL_PATH, 'img_label.csv')
label = pd.read_csv(label_path)
#label = label[:1280]

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

# model
#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(2)(x)
max = 1
min = -1
predictions = Lambda(lambda x: (x - min) / (max - min))(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# Training
optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)

def reg_loss_fn(y_true, y_pred):
    y_pred = tf.transpose(y_pred)
    loss = tf.keras.losses.MSE(y_true, y_pred)
    return tf.reduce_mean(loss)

@tf.function
def train_step(dataset):
    x, y = dataset
    image = x
    va_label = y
    loss = 0

    with tf.GradientTape() as cls_tape:
        decision = model(image, training=True)
        cls_loss = reg_loss_fn(va_label, decision)
        loss += cls_loss

    lotal_loss = loss

    # gradients
    gradients_of_img_classifier = cls_tape.gradient(cls_loss, model.trainable_variables)
    # optimizers
    optimizer.apply_gradients(zip(gradients_of_img_classifier, model.trainable_variables))

    return lotal_loss


score_plot = []

def train(dataset, epochs):
    for epoch in tqdm(range(epochs), desc="EPOCHS"):
        start = time.time()

        batches = 0
        with tqdm(total=len(dataset), desc="BATCHES") as pbar:
            for image_batch in dataset:
                reg_loss = train_step(image_batch)
                batches += 1
                pbar.update(1)

                if batches >= len(dataset):
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    pbar.update(1)
                    pbar.close()
                    break

        # save each step's loss
        score_plot.append(reg_loss)

        tqdm.write('Epoch {} Loss {:.6f}'.format(epoch + 1, reg_loss))
        tqdm.write('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

train(training_set, EPOCHS)

# 6. plot loss graphs
def clac_MSE(y_true, y_pred):
    y_pred = tf.transpose(y_pred)
    loss = tf.keras.losses.MSE(y_true, y_pred)
    return tf.reduce_mean(loss).numpy()

x, y = next(iter(validataion_set))
seed_img = x[:]
decision = model(seed_img, training=False)
score = clac_MSE(y, decision)

plt.plot(score_plot, label='Score')
plt.title("Average Score {:.6f}".format(score))
plt.xlabel('Epochs')
plt.ylabel('score')
plt.legend(loc='upper right')
#plt.savefig(os.path.join(PLT_PATH, 'resnet50_score.png'))
plt.savefig(os.path.join(PLT_PATH, 'vgg19_score.png'))
#plt.savefig(os.path.join(PLT_PATH, 'vgg16_score.png'))
