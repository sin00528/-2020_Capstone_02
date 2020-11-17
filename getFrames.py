import numpy as np
import cv2
import os
import skvideo.io
from tqdm import tqdm

IN_PATH = './video'
os.makedirs('./img_seq', exist_ok=True)
OUT_PATH = './img_seq'

def getFrames():
    for portion in os.listdir(IN_PATH):
        # import pdb; pdb.set_trace()
        # portion: train, val.
        path_portion = IN_PATH + "/" + portion
        for filename in tqdm(os.listdir(path_portion)):
            fullpath = path_portion + "/" + filename
            filename = int(filename.split('.')[0])
            vid = skvideo.io.vread(fullpath)
            #os.makedirs(os.path.join(OUT_PATH, portion, str(filename)), exist_ok=True)
            os.makedirs(os.path.join(OUT_PATH, portion, str('{:04d}'.format(int(filename)))), exist_ok=True)

            num_frame = 1
            for frame in vid:
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if img.size == 0 :
                    continue

                savePath = os.path.join(OUT_PATH, portion, str('{:04d}'.format(int(filename))), str('{:05d}'.format(str(num_frame))) + '.jpg')
                cv2.imwrite(savePath, img)
                num_frame += 1


def main():
    getFrames()

if __name__ == '__main__':
    main()