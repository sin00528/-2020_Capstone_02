import os
import numpy as np
import dlib
import cv2
import skvideo.io
from imutils import face_utils
import openface
from tqdm import tqdm

IN_PATH = './video'
os.makedirs('./img_seq_crop', exist_ok=True)
OUT_PATH = './img_seq_crop'

# load face detector & landmark predictor 
landmarker = "./dat/shape_predictor_68_face_landmarks.dat"
hog_detector = dlib.get_frontal_face_detector()
face_detector = dlib.cnn_face_detection_model_v1("./dat/mmod_human_face_detector.dat")
face_predictor = dlib.shape_predictor(landmarker)
face_aligner = openface.AlignDlib(landmarker)

def face_part_extract():
    for portion in os.listdir(IN_PATH):
        path_portion = IN_PATH + "/" + portion
        for filename in tqdm(os.listdir(path_portion)):
            fullpath = path_portion + "/" + filename
            filename = int(filename.split('.')[0])
            vid = skvideo.io.vread(fullpath)
            os.makedirs(os.path.join(OUT_PATH, portion, str('{:04d}'.format(int(filename)))), exist_ok=True)
            

            # dlib cnn face detector #1
            num_frame = 1
            for frame in vid:
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                if img.size == 0 :
                    continue

                num_frame += 1
                
                # face_roi
                rects = face_detector(img, 1)
                for i, det in enumerate(rects):
                    #print("Detection :", i)
                    #print("Confidence :", det.confidence)
                    #if det.confidence < 1:
                    #    continue 
                    l = det.rect.left()
                    t = det.rect.top()
                    r = det.rect.right()
                    b = det.rect.bottom()
                    
                    faceRect = det.rect
                    shape = face_utils.shape_to_np(face_predictor(gray, faceRect))

                    alignedFace = face_aligner.align(128, img, faceRect, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
                    savePath = os.path.join(OUT_PATH, portion, str('{:04d}'.format(int(filename))), str('{:05d}'.format(int(num_frame))) + '.jpg')
                    cv2.imwrite(savePath, alignedFace)
                    #num_frame += 1

def main():
    face_part_extract()

if __name__ == '__main__':
    main()