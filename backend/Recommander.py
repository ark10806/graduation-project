import heapq
from typing import *
from tqdm import tqdm

from models import Encoders
from utils import pickleIO, params


def recommand(msg: str) -> List[Tuple[str, float]]:
    clip = Encoders.MultiModalClip()
    meme_features = pickleIO.loadPkl(params.meme_feature_path)
    #? 전역변수로 바꿔야함
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