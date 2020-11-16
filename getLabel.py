import numpy as np
import os
import csv
import pandas as pd
from tqdm import tqdm

# for multiprocessing
from multiprocessing import Pool

META_PATH = './annotations/'
IN_PATH = './annotations/train/'
OUT_PATH = './annotations/'

meta_path = os.path.join(META_PATH, 'metafile.csv')
meta = pd.read_csv(meta_path)

# NOTE: ONLY TRAIN SET ANNOTATION PROVIDED
meta = meta[meta.set_name == 'train']

# NOTE: SMAll BATCH TEST
#meta = meta[:2]

# MAKE label.csv file header
df = pd.DataFrame([], columns=['file_name', 'num_frame', 'valence', 'arousal'])
outpath = os.path.join(OUT_PATH, 'label.csv')
df.to_csv(outpath, index=False, mode='w')

def getVA(idx):
    file_name = meta['file_name'][idx]
    
    # make local annotation dataframe
    df = pd.DataFrame([], columns=['file_name', 'num_frame', 'valence', 'arousal'])

    # load V/A values from annotations folder
    valence = []
    arousal = []

    file_name = file_name[:-4]
    fullpath = file_name + '.txt'
    
    v_path = os.path.join(IN_PATH, 'valence', fullpath)
    a_path = os.path.join(IN_PATH, 'arousal', fullpath)

    v_handler = open(v_path, 'r')
    a_handler = open(a_path, 'r')

    valence = v_handler.readlines()
    arousal = a_handler.readlines()

    # remove newline character & zip V/A
    valence = list(map(lambda s: s.strip(), valence))
    arousal = list(map(lambda s: s.strip(), arousal))
    lines = zip(valence, arousal)

    # NOTE: DataFrame append loop
    for num_frame, line in enumerate(lines):  
        row = pd.DataFrame([[file_name, num_frame+1, line[0], line[1]]], columns=['file_name', 'num_frame', 'valence', 'arousal'])
        df = df.append(row, ignore_index=True)

    # Append csv file
    outpath = os.path.join(OUT_PATH, 'label.csv')
    df.to_csv(outpath, mode='a', index=False, header=False)

    v_handler.close()
    a_handler.close()

# NOTE: multiprocssing
processes = os.cpu_count()
with Pool(processes=processes) as p:
    max_ = len(meta)
    with tqdm(total=max_) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(getVA, range(max_)))):
            pbar.update()
