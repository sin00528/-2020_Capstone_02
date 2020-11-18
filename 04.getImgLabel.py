import os
import pandas as pd
from tqdm import tqdm

import keras
from keras.losses import categorical_crossentropy
import keras.backend as K

LABEL_PATH = './annotations/'
IMG_PATH = './img_seq/'
OUT_PATH = './annotations/'

label_path = os.path.join(LABEL_PATH, 'label.csv')
label = pd.read_csv(label_path)

# MAKE img_label.csv
df = pd.DataFrame([], columns=['file_name', 'num_frame', 'valence', 'arousal'])

for portion in os.listdir(IMG_PATH):
    path_portion = IMG_PATH + "/" + portion
    for dir_name in tqdm(os.listdir(path_portion)):
        dir_path = path_portion + "/" + dir_name

        # exclude folders not in label.csv
        if not int(dir_name) in label['file_name'].unique():
            continue

        frames = []
        for file_name in os.listdir(dir_path):
            full_path = dir_path + "/" + file_name
            file_name = int(file_name.split('.')[0])

            # exclude img not in label.csv
            if not int(file_name) in label['num_frame']:
                continue

            frames.append(int(file_name))
            
        # exclude img not in label.csv
        row = label[label['file_name'].isin([int(dir_name)]) & label['num_frame'].isin(frames)]
        df = df.append(row, ignore_index=True)
    
    outpath = os.path.join(OUT_PATH, 'img_label.csv')
    df.to_csv(outpath, mode='w', index=False)