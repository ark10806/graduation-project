"""
1. Feature Extraction -> Pickle
2. VectorScan(Features) -> Meme
3. build dataset
"""
import os
import torch
import pickle
from typing import *
from tqdm import tqdm

import params
from models import Encoders
from utils import dataloader, pickleIO

categories = {
    'etc':          "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 
    'politic':      'It is about political issues or politicians in governments',
    'social':       'It is about social issues, pandemics or vaccine for COVID-19 virus or photos of family',
    'economy':      'It is about monetary economy, stocks or bitcoins', 
    'fashion':      'A photo of fashions, clothings, wearings or shoes',
    'entertain':    'A photo of musicians, entertainers, celebrity, music bands or idols', 
    'sport':        'A photo of sports or athletes', 
    'food':         'It is about dishes, foods, eating things', 
    'animal':       'A photo of pet animals or companion animals',
    'selfi':        'A photo of self-taken (portrait) photographs',
    'meme':         'A picture of memes or funny clips',
    'cartoon':      'A photo or picture of strip cartoons, comics, sketches or drawings', 
    'movie':        'It is about movies or cinema', 
    'game':         'A photo that has taken in video games',
    'landscape':    'A photo of landscape or natural scenery',
}

def getCategories(self, feature: torch.Tensor, categories: List[str]) -> float:
    
