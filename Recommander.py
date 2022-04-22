import torch
from model import Encoders
from utils import pickleIO, params
from typing import *


clip = Encoders.MultiModalClip()
meme_features = pickleIO.loadFeatures(params.meme_feature_path)
def recommand(msg: str) -> List[str]:
    txt_emb = clip.get_text_feature(msg)
    