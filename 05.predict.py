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

EPOCHS = 200
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
import dlib
import cv2
import skvideo.io
from imutils import face_utils
import openface

# 1. build models
img_encoder = make_encoder()
img_classifier = make_classifier(num_classes=2)

img_decoder = make_decoder()
img_discriminator = make_discriminator()

# 2. optimizers
img_classifier_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_decoder_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
img_discriminator_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)

# 3. restore ckpt
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(img_decoder_optimizer=img_decoder_optimizer,
                                img_discriminator_optimizer=img_discriminator_optimizer,
                                img_classifier_optimizer=img_classifier_optimizer,
                                img_encoder=img_encoder,
                                img_classifier=img_classifier,
                                img_decoder=img_decoder,
                                img_discriminator=img_discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# load face detector & landmark predictor 
landmarker = "./dat/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.cnn_face_detection_model_v1("./dat/mmod_human_face_detector.dat")
face_predictor = dlib.shape_predictor(landmarker)
face_aligner = openface.AlignDlib(landmarker)

os.makedirs('./demo/output', exist_ok=True)

IN_PATH = "./demo/input/demo_input.mp4"
OUT_PATH = "./demo/output/demo_output.mp4"
vid = skvideo.io.vread(IN_PATH)

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

"""
def emotion_txt(pred):
    # pred[0] : v
    # pred[1] : a
    emotion = ''
    if pred[0] > 0:
        if pred[1] > 0:
            emotion = 'happy'
        else:
            emotion = 'comfort'
    else:
        if pred[1] > 0:
            emotion = 'anxious'
        else:
            emotion = 'bored'
    return emotion
"""

frames = []
num_frame = 1
for frame in vid:
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img.size == 0 :
        continue

    num_frame += 1
    
    # face_roi
    rects = face_detector(img, 1)
    if len(rects) == 0:
        frames.append(rgb)

    for i, det in enumerate(rects):
        l = det.rect.left()
        t = det.rect.top()
        r = det.rect.right()
        b = det.rect.bottom()
        
        faceRect = det.rect
        shape = face_utils.shape_to_np(face_predictor(gray, faceRect))

        alignedFace = face_aligner.align(128, img, faceRect, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        aligned_face_image = np.expand_dims(prep_fn(alignedFace), axis=0)
        
        encoded_image = img_encoder(aligned_face_image, training=False)
        decision = img_classifier(encoded_image, training=False)

        # write bbox and v/a value every frames
        cv2.rectangle(rgb, (l, t), (r, b), (0, 255, 0), thickness=1)
        txt = '{}'.format(np.round(decision[0], 3))
        #txt = '{} : {}'.format(np.round(decision[0], 3), emotion_txt(decision[0]))
        cv2.putText(rgb, txt, (l, t), 0, 0.5, (255, 0, 0), thickness=2)
        frames.append(rgb)
        
skvideo.io.vwrite(OUT_PATH, frames)

