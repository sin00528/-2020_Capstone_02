import os
import time
import random
import numpy as np
import pandas as pd
import imageio
import glob
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
os.makedirs('./plt/', exist_ok=True)
PLT_PATH = './plt/'

EPOCHS = 20
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
#label = label[:1000]

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

# 2. build models
img_encoder = make_encoder()
img_classifier = make_classifier(num_classes=2)

img_decoder = make_decoder()
img_discriminator = make_discriminator()

# 3. define loss func and optimizers
# 3.1. loss func
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # loss helper func

def cls_loss_fn(y_true, y_pred):
    y_pred = tf.transpose(y_pred)
    loss = tf.keras.losses.MSE(y_true, y_pred)
    return tf.reduce_mean(loss)

def pixel_loss_fn(real_img, fake_img):
    l1_distance = K.abs(fake_img - real_img)
    loss = l1_distance / ((IMG_HEIGHT * IMG_WIDTH * IMG_CHANNEL * (BATCH_SIZE)) + K.epsilon()) 
    return loss

def d_loss_fn(real_img, fake_img):
    real_loss = cross_entropy(tf.ones_like(real_img), real_img)
    fake_loss = cross_entropy(tf.zeros_like(fake_img), fake_img)
    total_loss = real_loss + fake_loss
    return total_loss

# 3.2. optimizers
img_classifier_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_decoder_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_discriminator_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)

# 4. save ckpt
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(img_decoder_optimizer=img_decoder_optimizer,
                                img_discriminator_optimizer=img_discriminator_optimizer,
                                img_classifier_optimizer=img_classifier_optimizer,
                                img_encoder=img_encoder,
                                img_classifier=img_classifier,
                                img_decoder=img_decoder,
                                img_discriminator=img_discriminator)

# 4.1 load latest ckpt (just in case)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# set seed
x, _ = next(iter(validataion_set))
seed_img = x[:36]
seed = img_encoder(seed_img, training=False)

@tf.function
def train_step(dataset):
    x, y = dataset
    image = x
    va_label = y

    loss = 0
    
    with tf.GradientTape() as cls_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
        # encoders
        encoded_image = img_encoder(image, training=True)

        # classifiers
        decision = img_classifier(encoded_image , training=True)
        
        # decoders
        decoded_image = img_decoder(encoded_image, training=True)

        # discriminators
        image += 0.05 * np.random.random(image.shape)
        real_output = img_discriminator(image, training=True)
        fake_output = img_discriminator(decoded_image, training=True)

        # losses
        cls_loss = cls_loss_fn(va_label, decision)
        dec_loss = pixel_loss_fn(image, decoded_image)
        dis_loss = d_loss_fn(real_output, fake_output)

        loss += cls_loss

    lotal_loss = (loss / len(dataset))

    # gradients
    gradients_of_img_classifier = cls_tape.gradient(cls_loss, img_classifier.trainable_variables)
    gradients_of_img_decoder = dec_tape.gradient(dec_loss, img_decoder.trainable_variables)
    gradients_of_img_discriminator = disc_tape.gradient(dis_loss, img_discriminator.trainable_variables)

    # optimizers
    img_classifier_optimizer.apply_gradients(zip(gradients_of_img_classifier, img_classifier.trainable_variables))
    img_decoder_optimizer.apply_gradients(zip(gradients_of_img_decoder, img_decoder.trainable_variables))
    img_discriminator_optimizer.apply_gradients(zip(gradients_of_img_discriminator, img_discriminator.trainable_variables))

    return lotal_loss

cls_loss_plot = []

def train(dataset, epochs):
    for epoch in tqdm(range(epochs), desc="EPOCHS"):
        start = time.time()

        batches = 0
        with tqdm(total=len(dataset), desc="BATCHES") as pbar:
            for image_batch in dataset:
                cls_loss = train_step(image_batch)
                batches += 1
                pbar.update(1)

                if batches >= len(dataset):
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    pbar.update(1)
                    pbar.close()
                    break
        
        # save each step's loss
        cls_loss_plot.append(cls_loss)

        # save generated images
        generate_and_save_images(img_decoder, img_classifier, epoch + 1, seed)

        # save model every 5 EPOCH
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        tqdm.write('Epoch {} Loss {:.6f}'.format(epoch + 1, cls_loss))
        tqdm.write('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # save generated images (end of epoch)
    generate_and_save_images(img_decoder, img_classifier, epochs, seed)

def generate_and_save_images(dec_model, cls_model, epoch, test_input):
    predictions = dec_model(test_input, training=False)
    va_value = cls_model(test_input, training=False)

    fig = plt.figure()

    xlabels = ['{}'.format(np.round(va_value[i], 3)) for i in range(predictions.shape[0])]
    #import pdb; pdb.set_trace()

    for i in range(predictions.shape[0]):
        ax = fig.add_subplot(6, 6, i+1)
        ax.imshow(np.uint8(predictions[i, :, :, :] * 127.5 + 127.5))
        ax.set_xlabel(xlabels[i], fontsize=5)
        ax.set_xticks([]), ax.set_yticks([])
        #plt.axis('off')

    fig.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()
    #plt.show()

# 5. model train
# 5.1 send train start msg to discord channel
import json
with open('./secrets.json') as f:
    key_file = json.loads(f.read())

from discord_webhook import DiscordWebhook
url = key_file["DISCORD_URL"]
webhook = DiscordWebhook(url=url, content='Train Started...')
response = webhook.execute()

# 5.2 model train
try:
    train(training_set, EPOCHS)
except:
    print('error occured')
    webhook = DiscordWebhook(url=url, content='An error has occured while training...')
    response = webhook.execute()

# 5.1 send train finished msg to discord channel
webhook = DiscordWebhook(url=url, content='Train Finished...')
response = webhook.execute()

# 6. plot loss graphs
plt.plot(cls_loss_plot, label='cls_loss')
plt.title("Loss graph")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig(os.path.join(PLT_PATH, 'gan_loss.png'))

# 7. gen gif
anim_file = os.path.join(OUT_PATH, 'gan.gif') 

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(os.path.join(OUT_PATH, 'image*.png'))
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
