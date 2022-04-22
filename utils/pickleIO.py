import os
import torch
import pickle
from typing import *
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models import Encoders
from utils import dataloader, params


def saveFeatures(data_root: str, pkl_file: str):
    dataset = dataloader.ImageTextPair(data_root)
    output = []
    failed = 0
    for image, label in tqdm(dataset, desc='Extr features'):
        try:
            output.append({
                'fname':    label['fname'],
                'text':     label['text'],
                'feature':  clip.get_multimodal_feature(image, label['text'])
            })
        except Exception as e:
            failed += 1
    print(f'save {pkl_file}...', end='  ')
    with open(os.path.join(res_path, pkl_file), 'wb') as f:
        pickle.dump(output, f)
    print('complete!')
    print(f'loss rate: {failed / len(dataset) * 100: .2f}%')

def savePkl(data, pkl_file: str):
    print(f'save {pkl_file}...', end='  ')
    with open(os.path.join(params.res_path, pkl_file), 'wb') as f:
        pickle.dump(data, f)
    print('complete!')
    

def loadFeatures(pkl_file: str)-> List[torch.Tensor]:
    print(f'loading {pkl_file}..', end='  ')
    with open(os.path.join(params.res_path, pkl_file), 'rb') as f:
        pkl_data = pickle.load(f)
    print(f'[complete: {len(pkl_data)}]')
    return pkl_data




if __name__ == '__main__':
    data_root = params.data_root
    pkl_file = params.pkl_file
    res_path = params.res_path

    clip = Encoders.MultiModalClip()
    saveFeatures(data_root, pkl_file)