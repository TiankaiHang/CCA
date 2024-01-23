import os
import sys
import click

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import clip
import numpy as np

from einops import rearrange
from typing import List

import PIL
from PIL import Image
from tqdm import tqdm

# import ImageReward as RM
import requests


class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = MLP(768)
        # torch load checkpoint from url
        # wget "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth" -O "../sac+logos+ava1-l14-linearMSE.pth"
        if not os.path.exists("../sac+logos+ava1-l14-linearMSE.pth"):
            os.system('wget "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth" -O "../sac+logos+ava1-l14-linearMSE.pth"')
        s = torch.load("../sac+logos+ava1-l14-linearMSE.pth", map_location="cpu")
        self.model.load_state_dict(s)
        # self.model.to(self.device)
        self.model.eval()

        self.model2, self.preprocess = clip.load("ViT-L/14", device="cpu")  #RN50x64  

    def predict(self, img_path):
        if isinstance(img_path, str):
            if img_path.startswith("http"):
                pil_image = Image.open(requests.get(img_path, stream=True).raw)
            else:
                pil_image = Image.open(img_path)
        else:
            pil_image = img_path
        
        image = self.preprocess(pil_image).unsqueeze(0).cuda()

        with torch.no_grad():
            image_features = self.model2.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).cuda().type(torch.cuda.FloatTensor))

        return prediction.item()


def get_aesthetic_score(img_path):
    predictor = AestheticPredictor().cuda()
    return predictor.predict(img_path)


if __name__ == "__main__":
    predictor = AestheticPredictor().cuda()
    print(predictor.predict(
        "https://i.pinimg.com/736x/a7/64/f2/a764f2dc30ecf11641a6c422c652b57e.jpg"
    ))