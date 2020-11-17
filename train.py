import os
import pandas as pd
from tqdm import tqdm

# NOTE: load annotaion sample
LABEL_PATH = './annotations'

label_path = os.path.join(LABEL_PATH, 'label.csv')
label = pd.read_csv(label_path)
print(label[label['file_name'] == 118])

# NOTE: load face img sample
IMG_PATH = './img_seq_crop'

for portion in os.listdir(IMG_PATH):
    path_portion = IMG_PATH + "/" + portion
    for dir_name in tqdm(os.listdir(path_portion)):
        dir_path = path_portion + "/" + dir_name
        for file_name in os.listdir(dir_path):
            full_path = dir_path + "/" + file_name
            file_name = int(file_name.split('.')[0])
            print(full_path)
