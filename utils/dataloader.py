import os
import PIL
from typing import *
import pandas as pd
from PIL import Image

class ImageTextPair:
    def __init__(self, data_root: str):
        df = pd.read_csv(os.path.join(data_root, 'label.csv'))
        self.data_root = data_root
        self.fname = df['filename'].to_list()
        self.label = df['text'].to_list()
        del df
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx: int) -> Tuple:
        image = Image.open(os.path.join(self.data_root, 'images', self.fname[idx]))
        text = self.label[idx]
        return image, text