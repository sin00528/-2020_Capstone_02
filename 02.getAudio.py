import os
import moviepy.editor as mp
from tqdm import tqdm

IN_PATH = './video'
os.makedirs('./aud_seq', exist_ok=True)
OUT_PATH = './aud_seq'

for portion in os.listdir(IN_PATH):
    path_portion = IN_PATH + "/" + portion
    for filename in tqdm(os.listdir(path_portion)):
        fullpath = path_portion + "/" + filename
        filename = int(filename.split('.')[0])
        clip = mp.VideoFileClip(fullpath)
        os.makedirs(os.path.join(OUT_PATH, portion), exist_ok=True)
        savePath = os.path.join(OUT_PATH, portion, str('{:04d}'.format(int(filename)))) + '.mp3'
        print(savePath)
        clip.audio.write_audiofile(savePath)
