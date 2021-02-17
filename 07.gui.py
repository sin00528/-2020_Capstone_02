import sys
import numpy as np
import cv2

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5 import uic

form_class = uic.loadUiType("main_window.ui")[0]

# A VideoThread class which uses OpenCv
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
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

        #Input GroupBox안에 있는 RadioButton에 기능 연결
        self.radioButton_1.clicked.connect(self.groupRadFunction)
        self.radioButton_2.clicked.connect(self.groupRadFunction)

        #Output GroupBox안에 있는 CheckBox에 기능 연결
        self.checkBox_1.stateChanged.connect(self.groupchkFunction)
        self.checkBox_2.stateChanged.connect(self.groupchkFunction)
        self.checkBox_3.stateChanged.connect(self.groupchkFunction)
        self.checkBox_4.stateChanged.connect(self.groupchkFunction)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.IgnoreAspectRatio)
        #p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

    def groupRadFunction(self):
        if self.radioButton_1.isChecked() : print("rad_1 isChecked")
        elif self.radioButton_2.isChecked() : print("rad_2 isChecked")

    def groupchkFunction(self):
        # CheckBox는 여러개가 선택될 수 있기 때문에 elif를 사용하지 않습니다.
        if self.checkBox_1.isChecked() : print("chk_1 isChecked")
        if self.checkBox_2.isChecked() : print("chk_2 isChecked")
        if self.checkBox_3.isChecked() : print("chk_3 isChecked")
        if self.checkBox_4.isChecked() : print("chk_4 isChecked")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()