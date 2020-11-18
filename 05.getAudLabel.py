import os
import pandas as pd
from tqdm import tqdm

import keras
from keras.losses import categorical_crossentropy
import keras.backend as K

LABEL_PATH = './annotations/'
IMG_PATH = './aud_seq/'
OUT_PATH = './annotations/'

label_path = os.path.join(LABEL_PATH, 'label.csv')
label = pd.read_csv(label_path)

# MAKE img_label.csv
df = pd.DataFrame([], columns=['file_path', 'file_name', 'num_frame', 'valence', 'arousal'])

for portion in os.listdir(IMG_PATH):
    path_portion = os.path.join(IMG_PATH, portion)
    for file_name in tqdm(os.listdir(path_portion)):
        full_path =  os.path.join(path_portion, file_name)
        file_name = int(file_name.split('.')[0])

        # exclude folders not in label.csv
        if not int(file_name) in label['file_name'].unique():
            continue
        
        # exclude img not in label.csv
        row = label[label['file_name'].isin([int(file_name)])]
        row['file_path'] = full_path
        df = df.append(row, ignore_index=True)
        #import pdb; pdb.set_trace()
    
outpath = os.path.join(OUT_PATH, 'aud_label.csv')
df.to_csv(outpath, mode='w', index=False)