# model
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
from keras.optimizers import Adam
from model import make_encoder, make_decoder, make_discriminator, make_classifier

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

def prep_fn(img):
    """ pixel scaling func [-1, 1] """
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

# GUI
import sys
import numpy as np
import cv2

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5 import uic

form_class = uic.loadUiType("main_window.ui")[0]

class VideoThread(QThread):
    """ Class for CAM option """
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Emotion Recognition demo")
        self.disply_width = 1280
        self.display_height = 720
        self.flags = [0, 0, 0, 0]

        #Input GroupBox안에 있는 RadioButton에 기능 연결
        self.radioButton_1.clicked.connect(self.groupRadFunction)   # CAM
        self.radioButton_2.clicked.connect(self.groupRadFunction)   # File

        #Output GroupBox안에 있는 CheckBox에 기능 연결
        self.checkBox_1.stateChanged.connect(self.groupchkFunction) # Facial Landmarks Option
        self.checkBox_2.stateChanged.connect(self.groupchkFunction) # Bounding Box Option
        self.checkBox_3.stateChanged.connect(self.groupchkFunction) # 7 Emotions Option
        self.checkBox_4.stateChanged.connect(self.groupchkFunction) # VA Value Option

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""

        img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # face_roi
        rects = face_detector(img, 1)

        for i, det in enumerate(rects):
            l, t, r, b = det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom()
            faceRect = det.rect
            shape = face_utils.shape_to_np(face_predictor(gray, faceRect))

            alignedFace = face_aligner.align(128,
                                             img,
                                             faceRect,
                                             landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
            aligned_face_image = np.expand_dims(prep_fn(alignedFace), axis=0)

            encoded_image = img_encoder(aligned_face_image, training=False)
            decision = img_classifier(encoded_image, training=False)

            # NOTE : Facial Landmarks OPTION; Connect to checkBox_1
            if self.flags[0] == 1:
                rgb = self.facial_landmarks(img, shape)

            # NOTE : Bounding Boxs OPTION; Connect to checkBox_2
            # write bbox and v/a value every frames
            if self.flags[1] == 1:
                rgb = self.bounding_Boxs(rgb, l, t, r, b)

            # NOTE : 7 Emotions OPTION; Connect to checkBox_3
            if self.flags[2] == 1:
                 self.seven_emotions()

            # NOTE : VA value OPTION; Connect to checkBox_4
            if self.flags[3] == 1:
                rgb = self.va_value(rgb, l, t, decision)

        #qt_img = self.convert_cv_qt(cv_img)
        qt_img = self.convert_cv_qt(rgb)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.IgnoreAspectRatio)
        # p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

    def facial_landmarks(self, img, shape):
        # Facial Landmarks OPTION
        for j in range(68):
            x, y = shape[j, 0], shape[j, 1]
            img = cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            ret_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ret_rgb

    def bounding_Boxs(self, rgb, l, t, r, b):
        ret_rgb = cv2.rectangle(rgb, (l, t), (r, b), (0, 255, 0), thickness=1)
        return ret_rgb

    def seven_emotions(self):
        # TODO : Implement this code
        pass

    def va_value(self, rgb, l, t, decision):
        txt = '{}'.format(np.round(decision[0], 3))
        ret_rgb = cv2.putText(rgb, txt, (l, t), 0, 1, (255, 0, 0), thickness=2)
        return ret_rgb

    # Inputs
    # CameraOption
    def rad_1Selected(self):
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    # FileOption
    def rad_2Selected(self):
        print("rad2")
        fname = QFileDialog.getOpenFileName(self,
                                             'Open file',
                                             '/home',
                                             'All files (*.*)',
                                             options=QFileDialog.DontUseNativeDialog)
        self.image_label.setText(fname[0])

        import datetime
        basename = "output"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

        IN_PATH = fname
        OUT_PATH = os.path.join(os.path.dirname(fname[0]), "_".join([basename, suffix])) + '.mp4'
        #print(OUT_PATH)
        
        vid = skvideo.io.vread(IN_PATH[0])

        frames = []
        num_frame = 1
        for frame in vid:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img.size == 0:
                continue

            num_frame += 1

            # face_roi
            rects = face_detector(img, 1)
            if len(rects) == 0:
                frames.append(rgb)

            for i, det in enumerate(rects):
                l, t, r, b = det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom()
                faceRect = det.rect
                shape = face_utils.shape_to_np(face_predictor(gray, faceRect))

                alignedFace = face_aligner.align(128,
                                                 img,
                                                 faceRect,
                                                 landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
                aligned_face_image = np.expand_dims(prep_fn(alignedFace), axis=0)

                encoded_image = img_encoder(aligned_face_image, training=False)
                decision = img_classifier(encoded_image, training=False)

                # NOTE : Facial Landmarks OPTION; Connect to checkBox_1
                if self.flags[0] == 1:
                    rgb = self.facial_landmarks(img, shape)

                # NOTE : Bounding Boxs OPTION; Connect to checkBox_2
                if self.flags[1] == 1:
                    rgb = self.bounding_Boxs(rgb, l, t, r, b)

                # NOTE : 7 Emotions OPTION; Connect to checkBox_3
                if self.flags[2] == 1:
                    rgb = self.seven_emotions()

                # NOTE : VA value OPTION; Connect to checkBox_4
                if self.flags[3] == 1:
                    rgb = self.va_value(rgb, l, t, decision)

                frames.append(rgb)
        skvideo.io.vwrite(OUT_PATH, frames)


    def groupRadFunction(self):
        if self.radioButton_1.isChecked(): self.rad_1Selected()
        elif self.radioButton_2.isChecked(): self.rad_2Selected()

    def groupchkFunction(self):
        # CheckBox는 여러개가 선택될 수 있기 때문에 elif를 사용하지 않습니다.
        self.flags = [0, 0, 0, 0]
        if self.checkBox_1.isChecked(): self.flags[0] = 1
        if self.checkBox_2.isChecked(): self.flags[1] = 1
        if self.checkBox_3.isChecked(): self.flags[2] = 1
        if self.checkBox_4.isChecked(): self.flags[3] = 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()