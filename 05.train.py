import os
import pandas as pd
from tqdm import tqdm
import keras

# NOTE: load annotaion sample
LABEL_PATH = './annotations/'
IMG_PATH = './img_seq/'

label_path = os.path.join(LABEL_PATH, 'img_label.csv')
label = pd.read_csv(label_path)

print(label)

"""
for portion in os.listdir(IMG_PATH):
    path_portion = IMG_PATH + "/" + portion
    for dir_name in tqdm(os.listdir(path_portion)):
        dir_path = path_portion + "/" + dir_name
        for file_name in os.listdir(dir_path):
            full_path = dir_path + "/" + file_name
            file_name = int(file_name.split('.')[0])
            print(full_path)
"""