import torch
from model import Encoders
from utils import pickleIO, params
from typing import *


clip = Encoders.MultiModalClip()
meme_features = pickleIO.loadFeatures(params.meme_feature_path)
def recommand(msg: str) -> List[str]:
    """ input message language must be English words. """
    msg_emb = clip.get_text_feature(msg)
    bag_of_memes = []
    for info in meme_features:
        _, similiarity = clip.inference(info['feature'], msg_emb)
        similiarity = similiarity[0]
        bag_of_memes.append( (similiarity, info['img_link']) )