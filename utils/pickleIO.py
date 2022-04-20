import os
import torch
import pickle
from typing import *
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import params
from models import Encoders
from utils import dataloader


def saveFeatures(data_root: str, pkl_file: str):
    dataset = dataloader.ImageTextPair(data_root)
    output = []
    failed = 0
    for img, txt in tqdm(dataset, desc='Extr features'):
        try:
            output.append(clip.get_multimodal_feature(img, txt))
        except Exception as e:
            failed += 1
    print(f'save {pkl_file}...', end='  ')
    with open(os.path.join(res_path, pkl_file), 'wb') as f:
        pickle.dump(output, f)
    print('complete!')
    print(f'loss rate: {failed / len(dataset) * 100: .2f}%')

def loadFeatures(pkl_file: str)-> List[torch.Tensor]:
    print(f'loading {pkl_file}..', end='  ')
    with open(os.path.join('./result', pkl_file)) as f:
        features = pickle.load(f)
    print(f'[complete: {len(features)}]')
    return features


if __name__ == '__main__':
    data_root = params.data_root
    pkl_file = params.pkl_file
    res_path = params.res_path

    clip = Encoders.MultiModalClip()
    saveFeatures(data_root, pkl_file)