import heapq
from typing import *
from tqdm import tqdm

from model import Encoders
from utils import pickleIO, params


clip = Encoders.MultiModalClip()
meme_features = pickleIO.loadFeatures(params.meme_feature_path)
def recommand(msg: str) -> List[Tuple[str, float]]:
    """ input message language must be English words. """
    msg_emb = clip.get_text_feature(msg)
    bag_of_memes = []
    for info in tqdm(meme_features, desc='recommand'):
        _, similiarity = clip.inference(info['feature'], msg_emb)
        similiarity = similiarity[0]
        heapq.heappush(bag_of_memes, (-similiarity, info['image_url']))
        output = []
        for _ in range(params.topk):
            sim, img_url = heapq.heappop(bag_of_memes)
            output.append({
                'image_url': img_url,
                'similarity': -sim
            })
        del bag_of_memes
    return output