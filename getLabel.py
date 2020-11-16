import numpy as np
import os
import csv
import pandas as pd
from tqdm import tqdm

META_PATH = './dat/'
IN_PATH = './annotations/train'
OUT_PATH = './annotations/'

def getLabel():
    meta_path = os.path.join(META_PATH, 'metafile.csv')
    meta = pd.read_csv(meta_path)

    # NOTE : ONLY train set annotation provided
    meta = meta[meta.set_name == 'train']

    # make annotation dataframe
    df = pd.DataFrame([], columns=['file_name', 'num_frame', 'valence', 'arousal'])

    for file_name in tqdm(meta['file_name']):
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

        # remove newline char
        valence = list(map(lambda s: s.strip(), valence))
        arousal = list(map(lambda s: s.strip(), arousal))

        lines = zip(valence, arousal)

        for num_frame, line in enumerate(lines):
            row = pd.DataFrame([[file_name, num_frame+1, line[0], line[1]]], columns=['file_name', 'num_frame', 'valence', 'arousal'])
            df = df.append(row)
            #print(df)
            #import pdb; pdb.set_trace()

        v_handler.close()
        a_handler.close()
    
    outpath = os.path.join(OUT_PATH, 'label.csv')
    df.to_csv(outpath, mode='w')


def main():
    getLabel()


if __name__ == '__main__':
    main()