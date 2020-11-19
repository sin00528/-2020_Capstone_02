import os
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

EPOCHS = 100
RND_SEED = 42
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128

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

print(label)

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)

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
img_id_encoder = make_encoder() # loss : cls_loss, dispel_loss
img_exp_encoder = make_encoder() # loss : cls_loss, dispel_loss

img_id_classifier = make_classifier(num_classes=1) # loss : cls_loss, dispel_loss
img_exp_classifier = make_classifier(num_classes=2) # loss : cls_loss, dispel_loss

img_decoder = make_decoder() # loss : adv_loss

img_discriminator = make_discriminator() # loss : pixel_loss

# 2.1. check encoder (Before training)
noise = tf.random.normal([1, 128, 128, 3])
encoded_id_image = img_id_encoder(noise, training=False)
encoded_exp_image = img_exp_encoder(noise, training=False)

print("encoded_id_image shape: ", encoded_id_image.shape)
print("encoded_exp_image shape: ", encoded_exp_image.shape)

# concat id & exp tensor
encoded_image = tf.concat([encoded_id_image, encoded_exp_image], axis=0)

# 2.2. check generated img (Before training)
generated_image = img_decoder(encoded_image, training=False)

print("generated_image shape: ", generated_image.shape)

plt.imsave(OUT_PATH + 'gen_id.png', generated_image[0, :, :, 0])
plt.imsave(OUT_PATH + 'gen_exp.png', generated_image[1, :, :, 0])

# 2.3 check discriminator works (Before training)
# NOTE : discriminator model returns pos value(real), neg value(fake)
#decision = discriminator(generated_image)
decision_id = img_id_classifier(encoded_id_image)
decision_exp = img_exp_classifier(encoded_exp_image)

print("decision_id shape: ", decision_id)
print("decision_exp shape: ", decision_exp)

# 3. define loss func and optimizers
# 3.1. loss func
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # loss helper func

def cls_loss(y_true, y_pred):
    loss = cross_entropy(y_true, y_pred)
    return loss

def dispel_loss(y_true, y_pred):
    loss = cross_entropy(y_true, y_pred)
    return loss

def pixel_loss(x_real, x_gen):
    l1_distance = 0
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            l1_distance += K.abs(x_gen[i][j] - x_real[i][j])

    loss = l1_distance / (IMG_HEIGHT * IMG_WIDTH)
    
    return loss

def adv_loss(y_true, y_pred):
    # wasserstein_loss
    loss = K.mean(y_true * y_pred)
    return loss

"""
def discriminator_loss(x_real, x_gen):
    real_loss = cross_entropy(tf.ones_like(x_real), x_real)
    fake_loss = cross_entropy(tf.zeros_like(x_gen), x_gen)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
"""

# 3.2. optimizers
img_exp_encoder_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_decoder_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_discriminator_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_classifier_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)


# 4. save ckpt
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(img_exp_encoder_optimizer=img_exp_encoder_optimizer,
                                img_decoder_optimizer=img_decoder_optimizer,
                                img_discriminator_optimizer=img_discriminator_optimizer,
                                img_classifier_optimizer=img_classifier_optimizer,
                                img_id_encoder=img_id_encoder,
                                img_exp_encoder=img_exp_encoder,
                                img_id_classifier=img_id_classifier,
                                img_exp_classifier=img_exp_classifier,
                                img_decoder=img_decoder,
                                img_discriminator=img_discriminator)


# TODO : make this code work
# 4. define train loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
        train_step(image_batch)

    # GIF를 위한 이미지를 바로 생성합니다.
    display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, seed)

    # 15 에포크가 지날 때마다 모델을 저장합니다.
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # 마지막 에포크가 끝난 후 생성합니다.
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# 5. model train
train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""
# 6. gen gifs
# 에포크 숫자를 사용하여 하나의 이미지를 보여줍니다.
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
"""
