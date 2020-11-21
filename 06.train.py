import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import keras
import keras.backend as K
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Input, Reshape, LeakyReLU, ZeroPadding2D, Activation, Add
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, Conv2DTranspose
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam, RMSprop, SGD

from model import make_encoder, make_decoder, make_discriminator, make_classifier


# NOTE: load annotaion sample
LABEL_PATH = './annotations/'
IMG_PATH = './img_seq/train/'
os.makedirs('./gan_images/', exist_ok=True)
OUT_PATH = './gan_images/'

EPOCHS = 20
RND_SEED = 42
BATCH_SIZE = 128
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
#label = label[:1280]

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

train_datagen = ImageDataGenerator(preprocessing_function=prep_fn, validation_split=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_dataframe(label,
                                                x_col = 'file_path',
                                                y_col= ['file_name', 'valence', 'arousal'],
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                class_mode="multi_output",
                                                subset='training')


validataion_set = train_datagen.flow_from_dataframe(label,
                                                x_col = 'file_path',
                                                y_col= ['file_name', 'valence', 'arousal'],
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                class_mode="multi_output",
                                                subset='validation')

"""
# NOTE : FOR TEST
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_dataframe(label,
                                            x_col = 'file_path',
                                            y_col= ['valence', 'arousal'],
                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                            batch_size=BATCH_SIZE,
                                            class_mode="multi_output")
"""

# 2. build models
img_id_encoder = make_encoder()
img_exp_encoder = make_encoder()

img_id_classifier = make_classifier(num_classes=1)
img_exp_classifier = make_classifier(num_classes=2)

img_decoder = make_decoder()

img_discriminator = make_discriminator()

"""
# 2.1. check encoder (Before training)
noise = tf.random.normal([1, 128, 128, 3])
encoded_id_image = img_id_encoder(noise, training=False)
encoded_exp_image = img_exp_encoder(noise, training=False)

print("encoded_id_image shape: ", encoded_id_image.shape)
print("encoded_exp_image shape: ", encoded_exp_image.shape)

# concat id & exp tensor
encoded_image = tf.concat([encoded_id_image, encoded_exp_image], axis=0)

# 2.2. check generated img (Before training)
decoded_image = img_decoder(encoded_image, training=False)

print("decoded_image shape: ", decoded_image.shape)

plt.imsave(OUT_PATH + 'dec_id.png', decoded_image[0, :, :, 0])
plt.imsave(OUT_PATH + 'dec_exp.png', decoded_image[1, :, :, 0])

# 2.3 check classifier works (Before training)
# NOTE : classifier model returns pos value(real), neg value(fake)
decision_id = img_id_classifier(encoded_id_image)
decision_exp = img_exp_classifier(encoded_exp_image)

print("decision_id shape: ", decision_id)
print("decision_exp shape: ", decision_exp)

# 2.4 check discriminator works (Before training)
# NOTE : discriminator provides the reconstruction loss
real_fake_tensor = img_discriminator(decoded_image, training=False)
print("real_fake_tensor shape: ", real_fake_tensor.shape)
"""

# 3. define loss func and optimizers
# 3.1. loss func
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # loss helper func

def cls_id_loss_fn(y_true, y_pred):
    #import pdb; pdb.set_trace()
    loss = cross_entropy(y_true, y_pred)
    return loss

def cls_exp_loss_fn(y_true, y_pred):
    #import pdb; pdb.set_trace()
    y_pred = tf.transpose(y_pred)
    loss = cross_entropy(y_true, y_pred)
    return loss

# def enc_id_loss_fn(y_true, y_pred):
#     loss = cross_entropy(y_true, y_pred)
#     return loss

def pixel_loss_fn(real_img, fake_img):
    real_img = tf.concat([real_img, real_img], axis=0)
    #import pdb; pdb.set_trace()
    """
    l1_distance = 0
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            for k in range(IMG_CHANNEL):
                #import pdb; pdb.set_trace()
                l1_distance += K.abs(fake_img[:][i][j][k] - real_img[:][i][j][k])
    
    loss = l1_distance / ((IMG_HEIGHT * IMG_WIDTH * IMG_CHANNEL * (BATCH_SIZE * 2)) + K.epsilon()) 
    return loss
    """
    real_loss = cross_entropy(tf.ones_like(real_img), real_img)
    fake_loss = cross_entropy(tf.zeros_like(fake_img), fake_img)
    total_loss = real_loss + fake_loss
    return total_loss


def d_loss_fn(real_img, fake_img):
    real_loss = cross_entropy(tf.ones_like(real_img), real_img)
    fake_loss = cross_entropy(tf.zeros_like(fake_img), fake_img)
    total_loss = real_loss + fake_loss
    return total_loss

# 3.2. optimizers
img_id_classifier_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_exp_classifier_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)

img_decoder_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_discriminator_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)

# 4. save ckpt
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(img_decoder_optimizer=img_decoder_optimizer,
                                img_discriminator_optimizer=img_discriminator_optimizer,
                                img_id_classifier_optimizer=img_id_classifier_optimizer,
                                img_exp_classifier_optimizer=img_exp_classifier_optimizer,
                                img_id_encoder=img_id_encoder,
                                img_exp_encoder=img_exp_encoder,
                                img_id_classifier=img_id_classifier,
                                img_exp_classifier=img_exp_classifier,
                                img_decoder=img_decoder,
                                img_discriminator=img_discriminator)

# set seed
#seed = tf.random.normal([16, 32, 32, 128])
#x, _ = next(iter(validataion_set))
x, _ = next(iter(training_set))
seed_img = x[:8]
seed_encoded_id = img_id_encoder(seed_img, training=False)
seed_encoded_exp = img_exp_encoder(seed_img, training=False)
seed = tf.concat([seed_encoded_id, seed_encoded_exp], axis=0)

@tf.function
def train_step(dataset):
    x, y = dataset
    image = x
    id_label = y[0]
    va_label = y[1:]
    
    with tf.GradientTape() as cls_id_tape, tf.GradientTape() as cls_exp_tape,\
        tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
        # encoders
        encoded_id_image = img_id_encoder(image, training=True)
        encoded_exp_image = img_exp_encoder(image, training=True)

        # classifiers
        decision_id = img_id_classifier(encoded_id_image, training=True)
        decision_exp = img_exp_classifier(encoded_exp_image, training=True)

        # concat encoded imgs
        encoded_image = tf.concat([encoded_id_image, encoded_exp_image], axis=0)
        
        # decoders
        decoded_image = img_decoder(encoded_image, training=True)

        # discriminators
        real_output = img_discriminator(image, training=True)
        fake_output = img_discriminator(decoded_image, training=True)

        # losses
        cls_id_loss = cls_id_loss_fn(id_label, decision_id)
        cls_exp_loss = cls_exp_loss_fn(va_label, decision_exp)

        dec_loss = pixel_loss_fn(image, decoded_image)
        dis_loss = d_loss_fn(real_output, fake_output)

    # gradients
    gradients_of_img_id_classifier = cls_id_tape.gradient(cls_id_loss, img_id_classifier.trainable_variables)
    gradients_of_img_exp_classifier = cls_exp_tape.gradient(cls_exp_loss, img_exp_classifier.trainable_variables)

    gradients_of_img_decoder = dec_tape.gradient(dec_loss, img_decoder.trainable_variables)
    gradients_of_img_discriminator = disc_tape.gradient(dis_loss, img_discriminator.trainable_variables)

    # optimizers
    img_id_classifier_optimizer.apply_gradients(zip(gradients_of_img_id_classifier, img_id_classifier.trainable_variables))
    img_exp_classifier_optimizer.apply_gradients(zip(gradients_of_img_exp_classifier, img_exp_classifier.trainable_variables))

    img_decoder_optimizer.apply_gradients(zip(gradients_of_img_decoder, img_decoder.trainable_variables))
    img_discriminator_optimizer.apply_gradients(zip(gradients_of_img_discriminator, img_discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in tqdm(range(epochs), desc="EPOCHS"):
        start = time.time()

        batches = 0
        for image_batch in tqdm(dataset, desc="BATCHES"):
            train_step(image_batch)
            batches += 1
            
            if batches >= len(dataset):
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        # save generated images
        generate_and_save_images(img_decoder, epoch + 1, seed)

        # save model each EPOCH
        checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # save generated images (end of epoch)
    generate_and_save_images(img_decoder, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.uint8(predictions[i, :, :, :] * 127.5 + 127.5))
        plt.axis('off')

    plt.savefig(os.path.join(OUT_PATH, 'image_at_epoch_{:04d}.png'.format(epoch)))
    #plt.show()

# 5. model train
train(training_set, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))