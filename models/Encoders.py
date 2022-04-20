import re
import torch
from typing import *
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class MultiModalClip:
    """ generates MultiModal Feature.
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.mark = re.compile('[^0-9|a-z|A-Z|\s|\.|\^|\$|\*|\+|\?|!|\\|@|#|\~|·_;:,|/|\-|>|<|%|…|─|&|\'|\(|\)]')
        self.web = re.compile("(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)")
        
    def norm(self, feature: torch.Tensor)-> torch.Tensor:
        return feature / feature.norm(dim=-1, keepdim=True)

    def text_preprocess(self, text: str) -> str:
        masks = (
            ('[^0-9|a-z|A-Z|\s|\.|\^|\$|\*|\+|\?|!|\\|@|#|\~|·_;:,|/|\-|>|<|%|…|─|&|\'|\(|\)]', ''),
            ('(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)', '<url>'),
            ('[\.]+', '.'),
            ('[,]+', ','),
            ('[~]+', '~'),
            ('[!]+', '!'),
            ('@\w+', '@someone'))
        for mask, stamp in masks:
            text = re.sub(mask, stamp, text)
        return text[:250]

    @torch.no_grad()
    def get_text_feature(self, text: str) -> torch.Tensor:
        if text is None: return None
        text = self.text_preprocess(text)
        text = self.processor(text=[text], return_tensors='pt', padding=True).to(self.device)
        txt_emb = self.clip.get_text_features(**text)
        return self.norm(txt_emb)

    @torch.no_grad()
    def get_image_feature(self, image: Image) -> torch.Tensor:
        if image is None: return None
        image = self.processor(images=image, return_tensors='pt', padding=True).to(self.device)
        img_emb = self.clip.get_image_features(**image)
        return self.norm(img_emb)

    @torch.no_grad()
    def get_multimodal_feature(self, image: Image=None, text: str=None, device: str='cpu') -> torch.Tensor:
        assert image is not None or text is not None, "needs <PIL.Image> or <str>"
        img_emb = self.get_image_feature(image).to(device)
        txt_emb = self.get_text_feature(text).to(device)
        if image is not None and text is not None:
            return torch.cat( (img_emb, txt_emb), dim=-1 )
        if image is not None:
            return img_emb
        if text is not None:
            return txt_emb
    
    def getCategories(self):
        pass


if __name__ == '__main__':
    strs = 'TC Sparkler Update Bracket play, Day 1 Shoutout to our two studs in the circle, @JaydenHeavener & @myaholt25 for throwing a combined no-hitter in our 9-0 win vs TN Mojo Mobley Bracket game #2 tomorrow @ 12p vs TN Mojo Hughes #boltsboom #boltsPremier2024 #L1nked'
    img = Image.open('/home/seungchan/Desktop/Projs/graduation-project/dataset/2022_02_all/images/2021_0.jpg')
    clip = MultiModalClip()
    multi_emb = clip.get_multimodal_feature(image=img, text=strs)
    print(multi_emb)