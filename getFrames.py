import numpy as np
import cv2
import os
from tqdm import tqdm

IN_PATH = './video'
os.makedirs('./Data', exist_ok=True)
OUT_PATH = './Data'

def getFrames():
    for portion in os.listdir(IN_PATH):
        # import pdb; pdb.set_trace()
        # portion: train, val.
        path_portion = IN_PATH + "/" + portion
        for filename in tqdm(os.listdir(path_portion)):
            fullpath = path_portion + "/" + filename
            filename = int(filename.split('.')[0])
            vidcap = cv2.VideoCapture(fullpath)
            #os.makedirs(os.path.join(OUT_PATH, portion, str(filename)), exist_ok=True)
            os.makedirs(os.path.join(OUT_PATH, portion, str('{:04d}'.format(int(filename)))), exist_ok=True)
            count = 1
            while (vidcap.isOpened()):
                isOK, image = vidcap.read()
                if(isOK):
                    #savePath = os.path.join(OUT_PATH, portion, str('{:04d}'.format(int(filename))), str(filename)+'_'+str(count)+'.jpg')
                    savePath = os.path.join(OUT_PATH, portion, str('{:04d}'.format(int(filename))), str(count) + '.jpg')
                    cv2.imwrite(savePath, image)
                    #print("Saved {}".format(savePath)))
                    isOK, image = vidcap.read()
                    count += 1
                else:
                    break
            vidcap.release()

def main():
    getFrames()

if __name__ == '__main__':
    main()