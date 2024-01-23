import os

import torch

from .edict_functions import EDICT_editing
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from .my_diffusers import AutoencoderKL, UNet2DConditionModel


def edict_api(img_path, base_prompt="", edit_prompt="", use_double=False, resolution=-1, use_p2p=True, seed=0):
    auth_token = ""
    # Build our CLIP model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
    clip = clip_model.text_model

    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)

    # Push to devices w/ double precision
    device = 'cuda'

    if use_double:
        unet.double().to(device)
        vae.double().to(device)
        clip.double().to(device)
    else:
        unet.float().to(device)
        vae.float().to(device)
        clip.float().to(device)

    # print("Loaded all models")

    _, edited_img = EDICT_editing(
        unet=unet, 
        vae=vae, 
        clip=clip, 
        clip_tokenizer=clip_tokenizer,
        im_path=img_path,
        base_prompt=base_prompt,
        edit_prompt=edit_prompt,
        run_baseline=False,
        init_image_strength=0.8,
        resolution=resolution,
        steps=50,
        use_p2p=use_p2p,
        use_double=use_double,
        seed=0)
    edited_img = edited_img[0]

    return edited_img