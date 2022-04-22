"""
1. Feature Extraction -> Pickle
2. VectorScan(Features) -> Meme
3. build dataset
"""
import os
import torch
import pickle
from typing import *
from PIL import Image
from tqdm import tqdm

from models import Encoders
from utils import dataloader, pickleIO, params

def getMeme():
    clip = Encoders.MultiModalClip()
    pkl_data = pickleIO.loadFeatures(params.pkl_file)
    output = []
    for data in tqdm(pkl_data):
        pred, prob = clip.getCategory(data['feature'], params.categories)
        if pred == 0: # Meme category
            output.append({
                'fname':    data['fname'],
                'text':     data['text'],
                'feature':  data['feature'],
                'prob':     prob[0]
            })
    
    pickleIO.savePkl(output, 'meme.pkl')
    print(f'len(meme): {len(output)}')

def visMeme():
    meme = pickleIO.loadFeatures('meme.pkl')
    for data in tqdm(meme, desc='vis memes'):
        img = Image.open(os.path.join(params.data_root, 'images', data['fname'])).resize((256,256))
        meme_path = os.path.join(params.res_path, 'meme')
        if not os.path.isdir(meme_path):
            os.makedirs(meme_path)
        img.save(os.path.join(meme_path, f'{data["prob"]*100:.2f}.jpeg' ))


if __name__ == '__main__':
    # getMeme()
    visMeme()