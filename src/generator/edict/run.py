import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pickle
import click
import numpy as np
import random
import json
from PIL import Image
import math

from tqdm.auto import tqdm
import requests
import clip
from einops import rearrange

from typing import Any, Dict, List, Optional, Tuple, Union

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

from .edict_functions import (
    EDICT_editing, 
    load_im_into_format_from_path,
    coupled_stablediffusion,
)

from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from .my_diffusers import AutoencoderKL, UNet2DConditionModel
from .my_diffusers.schedulers.scheduling_utils import SchedulerOutput
from .my_diffusers import LMSDiscreteScheduler, PNDMScheduler, DDPMScheduler, DDIMScheduler

# --------------------------------------------------------
# dist utils
# --------------------------------------------------------
def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def dist_init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    origin_size = None
    if not isinstance(data, torch.Tensor):
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
    else:
        origin_size = data.size()
        tensor = data.reshape(-1)

    tensor_type = tensor.dtype

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.FloatTensor(size=(max_size,)).cuda().to(tensor_type))
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size - local_size,)).cuda().to(tensor_type)
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            # suppose the difference of tensor size exist in first dimension
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list


# --------------------------------------------------------
# cal score
# --------------------------------------------------------
def load_image(image):
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        image = Image.open(image).convert("RGB")

    if isinstance(image, Image.Image):
        image = TF.to_tensor(image).unsqueeze(0)

    return image


class CLIPSimilarity(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(model_name, device="cpu")

    def get_similarity(self, image, text):
        if isinstance(image, str):
            if image.startswith("http"):
                image = Image.open(requests.get(image, stream=True).raw)
            else:
                image = Image.open(image)
            image = self.preprocess(image).unsqueeze(0).cuda()
        elif isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).cuda()
        elif isinstance(image, torch.Tensor):
            image = image.cuda()
        else:
            raise TypeError("image must be str or PIL.Image or torch.Tensor")

        text = clip.tokenize(text, truncate=True).cuda()

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)# .softmax(dim=-1)
        # values, indices = similarity[0].topk(5)

        return similarity[0]


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
        if img_path.startswith("http"):
            pil_image = Image.open(requests.get(img_path, stream=True).raw)
        else:
            pil_image = Image.open(img_path)
        image = self.preprocess(pil_image).unsqueeze(0).cuda()

        with torch.no_grad():
            image_features = self.model2.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).cuda().type(torch.cuda.FloatTensor))

        return prediction.item()


class CLIPSIM(nn.Module):

    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="~/cache/clip/")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text: List[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    # def forward(
    #     self, image_0: torch.Tensor, image_1: torch.Tensor, text_0: list[str], text_1: list[str]
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     image_features_0 = self.encode_image(image_0)
    #     image_features_1 = self.encode_image(image_1)
    #     text_features_0 = self.encode_text(text_0)
    #     text_features_1 = self.encode_text(text_1)
    #     sim_0 = F.cosine_similarity(image_features_0, text_features_0)
    #     sim_1 = F.cosine_similarity(image_features_1, text_features_1)
    #     sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
    #     sim_image = F.cosine_similarity(image_features_0, image_features_1)
    #     return sim_0, sim_1, sim_direction, sim_image

    def forward(self, image0, image1, text0, text1):
        image0 = load_image(image0).to(next(self.parameters()).device)
        image1 = load_image(image1).to(next(self.parameters()).device)
        image_features_0 = self.encode_image(image0)
        image_features_1 = self.encode_image(image1)
        text_features_0 = self.encode_text(text0)
        text_features_1 = self.encode_text(text1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image


@click.group()
def cli():
    r"""
    """


@cli.command()
# whether to use double
@click.option('--use_double', type=bool, default=False)
def test(use_double=False):
    # load model
    auth_token = ""
    # Build our CLIP model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip, cache_dir="../../.cache")
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16, cache_dir="../../.cache")
    clip = clip_model.text_model

    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")

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

    print("Loaded all models")
    
    im_path = "https://raw.githubusercontent.com/salesforce/EDICT/main/experiment_images/imagenet_camel.jpg"
    input_img, edited_img = EDICT_editing(
        unet=unet, 
        vae=vae, 
        clip=clip, 
        clip_tokenizer=clip_tokenizer,
        im_path=im_path,
        base_prompt='Camel by a fence with a sign',
        edit_prompt='Camel by a fence',
        run_baseline=False,
        init_image_strength=0.8,
        use_double=use_double)
    edited_img = edited_img[0]
    
    input_img.save("../logs/input_img.png")
    edited_img.save("../logs/edited_img.png")


# multiprocess
def main_worker(unet, vae, clip, clip_tokenizer, im_path, base_prompt, edit_prompt, instruction, out_path, init_image_strength=0.8, use_double=False):
    
    # im_path = "https://raw.githubusercontent.com/salesforce/EDICT/main/experiment_images/imagenet_camel.jpg"
    input_img, edited_img = EDICT_editing(
        unet=unet, 
        vae=vae, 
        clip=clip, 
        clip_tokenizer=clip_tokenizer,
        im_path=im_path,
        base_prompt=base_prompt,
        edit_prompt=edit_prompt,
        run_baseline=False,
        init_image_strength=init_image_strength,
        use_double=use_double)
    edited_img = edited_img[0]
    
    if out_path.endswith(".png"):
        input_img.save(out_path.replace(".png", "_input.png"))
        edited_img.save(out_path)
        instruction_path = out_path.replace(".png", "_instruction.txt")
    elif out_path.endswith(".jpg"):
        input_img.save(out_path.replace(".jpg", "_input.jpg"))
        edited_img.save(out_path)
        instruction_path = out_path.replace(".jpg", "_instruction.txt")
    else:
        raise ValueError(f"out_path must end with .png or .jpg, got {out_path}")
    with open(instruction_path, "w") as f:
        # write instruction, base_prompt, edit_prompt to file
        f.write("Instruction: " + instruction + "\n")
        f.write("Input: " + base_prompt + "\n")
        f.write("Output: " + edit_prompt + "\n")


@cli.command()
@click.option('--use-double', type=bool, default=False)
@click.option('--data-dir', type=str, default="../datasets")
@click.option('--init-image-strength', type=float, default=0.8)
@click.option('--seed', type=int, default=42)
def main(use_double=False, data_dir="../datasets", init_image_strength=0.8, seed=42):
    r"""
    """

    dist_init()

    save_dir = f"results-seed{seed}" if not use_double else f"results-seed{seed}_double"
    os.makedirs(save_dir, exist_ok=True)

    # set random seed
    SEED = seed + get_rank()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    meta_info_path = os.path.join(data_dir, "ordered_merged_results_new_v1.json")
    with open(meta_info_path, "r") as f:
        meta_info = json.load(f)

    # load model
    auth_token = ""
    # Build our CLIP model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip, cache_dir="../../.cache")
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16, cache_dir="../../.cache")
    clip = clip_model.text_model

    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")

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

    print0("Loaded all models")

    # img_paths = sorted(img_paths)
    meta_info = meta_info[get_rank()::get_world_size()]

    for idx, item in enumerate(meta_info):
        im_path = os.path.join(data_dir, item["input"])
        input_text = item["input_text"]
        output_text = item["output_text"]
        out_path = os.path.join(save_dir, im_path.replace(data_dir, "").replace("/", "__"))
        main_worker(
            unet=unet,
            vae=vae,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            im_path=im_path,
            instruction=item["instruction"],
            out_path=out_path,
            base_prompt=input_text,
            edit_prompt=output_text,
            init_image_strength=init_image_strength,
            use_double=use_double
        )


# -----------------------------------------------------------
# low level
# -----------------------------------------------------------
@cli.command()
@click.option('--use-double', type=bool, default=False)
@click.option('--data-dir', type=str, default="../datasets")
@click.option('--init-image-strength', type=float, default=0.8)
@click.option('--seed', type=int, default=42)
def main_low(use_double=False, data_dir="../datasets", init_image_strength=0.8, seed=42):
    r"""
    """

    dist_init()

    save_dir = f"logs/results-s{init_image_strength}-seed{seed}" if not use_double else f"results-s{init_image_strength}-seed{seed}_double"
    os.makedirs(save_dir, exist_ok=True)

    # set random seed
    SEED = seed + get_rank()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    meta_info_path = os.path.join(data_dir, "captions.json")
    with open(meta_info_path, "r") as f:
        meta_info = json.load(f)

    # load model
    auth_token = ""
    # Build our CLIP model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip, cache_dir="../../.cache")
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16, cache_dir="../../.cache")
    clip = clip_model.text_model

    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")

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

    print0("Loaded all models")

    # img_paths = sorted(img_paths)
    # meta_info = meta_info[get_rank()::get_world_size()]
    keys = list(meta_info.keys())
    # ends with jpg
    keys = [key for key in keys if key.endswith("jpg")]
    keys = sorted(keys)
    keys = keys[get_rank()::get_world_size()]

    for idx, key in enumerate(keys):
        im_path = os.path.join(data_dir, key.replace("../diffusion/data/", ""))
        if "gopro" or "deblur" in im_path.lower():
            input_text = meta_info[key] + ". It is a blurry image."
            output_text = input_text.replace("blurry", "sharp")
        elif "denoise" in im_path.lower():
            input_text = meta_info[key] + ". It is a noisy image."
            output_text = input_text.replace("noisy", "clean")
        elif "watermark" in im_path.lower():
            input_text = meta_info[key] + ". It is a watermarked image."
            output_text = input_text.replace("watermarked", "clean")
        
        out_path = os.path.join(save_dir, im_path.replace(data_dir, "").replace("/", "__"))
        main_worker(
            unet=unet,
            vae=vae,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            im_path=im_path,
            instruction="",
            out_path=out_path,
            base_prompt=input_text,
            edit_prompt=output_text,
            init_image_strength=init_image_strength,
            use_double=use_double
        )


@cli.command()
def cal_score():
    # data_path = "results_double"
    data_path = "results-seed1"

    dist_init()

    # rank 0 goes first
    if get_rank() != 0:
        torch.distributed.barrier()

    clip_sim = CLIPSimilarity().cuda()
    ap = AestheticPredictor().cuda()

    if get_rank() == 0:
        torch.distributed.barrier()

    files = os.listdir(data_path)
    txt_files = [f for f in files if f.endswith("_instruction.txt")]
    txt_files = sorted(txt_files)

    clip_scores = []
    ap_scores = []
    txt_files = txt_files[get_rank()::get_world_size()]

    for txt_file in tqdm(txt_files, total=len(txt_files), disable=get_rank() != 0):
        edited_img_path = os.path.join(data_path, txt_file.replace("_instruction.txt", ".png"))
        instruction_path = os.path.join(data_path, txt_file)
        output_caption = ""
        with open(instruction_path, "r") as f:
            instructions = f.readlines()
            # the line starts with "Instruction: "
            for line in instructions:
                if line.startswith("Output: "):
                    output_caption = line.replace("Output: ", "").strip()
                    break
        assert output_caption != "", "instruction is empty"
        # clip score
        clip_score = clip_sim.get_similarity(edited_img_path, output_caption).item()
        ap_score = ap.predict(edited_img_path)

        clip_scores.append(clip_score)
        ap_scores.append(ap_score)
            
    clip_scores = all_gather(clip_scores)
    ap_scores = all_gather(ap_scores)

    if get_rank() == 0:
        clip_scores = [item for sublist in clip_scores for item in sublist]
        ap_scores = [item for sublist in ap_scores for item in sublist]

        clip_scores = np.array(clip_scores)
        ap_scores = np.array(ap_scores)

        print("clip score: ", clip_scores.mean())
        print("ap score: ", ap_scores.mean())

    torch.distributed.barrier()



@cli.command()
@click.option('--data-dir', type=str, default="../datasets")
@click.option('--use-double', type=bool, default=False)
@click.option('--run-baseline', type=bool, default=False)
def invert_coco(data_dir="../datasets", use_double=False, run_baseline=False):
    import matplotlib.pyplot as plt

    dist_init()

    save_dir = "coco-inversion-edict" if not run_baseline else "coco-inversion-ddim"
    os.makedirs(save_dir, exist_ok=True)

    # load model
    auth_token = ""
    # Build our CLIP model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip, cache_dir="../../.cache")
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16, cache_dir="../../.cache")
    clip = clip_model.text_model

    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")

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

    print0("Loaded all models")

    # im = load_im_into_format_from_path('https://raw.githubusercontent.com/salesforce/EDICT/main/experiment_images/church.jpg')
    # load data
    json_path = os.path.join(data_dir, "coco", "annotations", "captions_val2017.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    annotation = data["annotations"]
    annotation = annotation[get_rank()::get_world_size()]

    for item in tqdm(annotation, total=len(annotation), disable=get_rank() != 0):
        im_path = os.path.join(data_dir, "coco", "val2017", f"{item['image_id']:012d}.jpg")
        im = load_im_into_format_from_path(im_path)
        prompt = item["caption"]

        latents = coupled_stablediffusion(
            unet=unet,
            vae=vae,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            init_image=im,
            prompt=prompt,
            run_baseline=run_baseline,
            reverse=True,
            use_double=use_double
        )

        if run_baseline:
            latents = latents[0]

        recon = coupled_stablediffusion(
            unet=unet,
            vae=vae,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            prompt=prompt,
            reverse=False,
            fixed_starting_latent=latents,
            run_baseline=run_baseline,
            use_double=use_double
        )
        recon = recon[0]

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(im)
        ax0.set_title("Original")
        ax1.imshow(recon)
        ax1.set_title("Recon")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{item['image_id']}_{item['id']}.png")


@cli.command()
@click.option('--use-double', type=bool, default=False)
@click.option('--run-baseline', type=bool, default=False)
def interpolate_test(use_double=False, run_baseline=False):

    dist_init()

    # set seed
    seed = 0 + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # load model
    auth_token = ""
    # Build our CLIP model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip, cache_dir="../../.cache")
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16, cache_dir="../../.cache")
    clip = clip_model.text_model

    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")

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

    latents1 = torch.randn(1, 4, 64, 64, device=device)
    latents2 = torch.randn(1, 4, 64, 64, device=device)
        
    prompts = [
        "a panda is eating bamboo",
    ]
    prompts = prompts[get_rank()::get_world_size()]
    num_splits = 2000
    for prompt in prompts:
        # for i in range(num_splits + 1):
        for i in range(980, 1001):
            # latents = latents1 + (latents2 - latents1) * (i / num_splits)
            latents = math.sqrt(1 - (i / num_splits)) * latents1 + math.sqrt(i / num_splits) * latents2
            recon = coupled_stablediffusion(
                unet=unet,
                vae=vae,
                clip=clip,
                clip_tokenizer=clip_tokenizer,
                prompt=prompt,
                reverse=False,
                fixed_starting_latent=latents,
                run_baseline=run_baseline,
                use_double=use_double
            )
            recon = recon[0]
            if not os.path.isdir(f"logs/{prompt.replace(' ', '_')}"):
                os.makedirs(f"logs/{prompt.replace(' ', '_')}", exist_ok=True)
            recon.save(f"logs/{prompt.replace(' ', '_')}/{num_splits}-{i}.png")


@cli.command()
@click.option('--data-dir', type=str, default="../../datasets/new_benchmarks2-july29/")
def image2gif(data_dir="../../datasets/new_benchmarks2-july29/"):
    import imageio
    import glob

    images = []
    for i in range(30):
        # images.append(imageio.imread(f"logs/a_dog_in_running_through_the_grass/{i}.png"))
        images.append(imageio.imread(f"logs/a_panda_is_eating_bamboo/{i}.png"))
    imageio.mimsave('logs/a_panda_is_eating_bamboo.gif', images, duration=200)



@cli.command()
@click.option('--data-dir', type=str, default="../datasets")
@click.option('--use-double', type=bool, default=False)
@click.option('--run-baseline', type=bool, default=False)
def aug28_invert_and_interpolate(data_dir="../datasets", use_double=False, run_baseline=False):
    import matplotlib.pyplot as plt
    import imageio

    dist_init()

    save_dir = "logs/coco-inversion-edict" if not run_baseline else "logs/coco-inversion-ddim"
    os.makedirs(save_dir, exist_ok=True)

    # load model
    auth_token = ""
    # Build our CLIP model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip, cache_dir=".cache")
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16, cache_dir=".cache")
    clip = clip_model.text_model

    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")

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

    print0("Loaded all models")

    # load video from url: https://github.com/TiankaiHang/storage-2023/releases/download/0828/stock-footage-billiards-concentrated-young-woman-playing-in-club.webm
    video_url = "https://github.com/TiankaiHang/storage-2023/releases/download/0828/stock-footage-billiards-concentrated-young-woman-playing-in-club.webm"
    save_path = os.path.join("logs", video_url.split("/")[-1])
    if not os.path.isfile(save_path):
        os.system(f"wget {video_url} -O {save_path}")
    reader = imageio.get_reader(save_path)
    fps = reader.get_meta_data()['fps']
    caption = video_url.split("/")[-1].replace(".webm", "").replace("-", " ")
    frames = []
    for frame in reader:
        frames.append(frame)
    frames = np.array(frames)
    im1 = frames[0]
    im2 = frames[-1]
    im1 = load_im_into_format_from_path(im1)
    im2 = load_im_into_format_from_path(im2)

    # import pdb; pdb.set_trace()

    latents1 = coupled_stablediffusion(
        unet=unet,
        vae=vae,
        clip=clip,
        clip_tokenizer=clip_tokenizer,
        init_image=im1,
        prompt=caption,
        run_baseline=run_baseline,
        reverse=True,
        use_double=use_double
    )

    latents2 = coupled_stablediffusion(
        unet=unet,
        vae=vae,
        clip=clip,
        clip_tokenizer=clip_tokenizer,
        init_image=im2,
        prompt=caption,
        run_baseline=run_baseline,
        reverse=True,
        use_double=use_double
    )

    if run_baseline:
        latents1 = latents1[0]
        latents2 = latents2[0]

    num_splits = 100
    fig, axes = plt.subplots(1, num_splits + 1, figsize=(5 * (num_splits + 1), 5))

    im1.save(f"logs/{caption.replace(' ', '_')}/src0.png")
    im2.save(f"logs/{caption.replace(' ', '_')}/src1.png")

    # recon1 = coupled_stablediffusion(
    #     unet=unet,
    #     vae=vae,
    #     clip=clip,
    #     clip_tokenizer=clip_tokenizer,
    #     prompt=caption,
    #     reverse=False,
    #     fixed_starting_latent=latents1,
    #     run_baseline=run_baseline,
    #     use_double=use_double
    # )

    # recon2 = coupled_stablediffusion(
    #     unet=unet,
    #     vae=vae,
    #     clip=clip,
    #     clip_tokenizer=clip_tokenizer,
    #     prompt=caption,
    #     reverse=False,
    #     fixed_starting_latent=latents2,
    #     run_baseline=run_baseline,
    #     use_double=use_double
    # )

    # recon1 = recon1[0]
    # recon2 = recon2[0]
    # recon1.save(f"logs/{caption.replace(' ', '_')}/recon0.png")
    # recon2.save(f"logs/{caption.replace(' ', '_')}/recon1.png")

    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    for i in range(num_splits + 1):
        latents = [
            math.sqrt(1 - (i / num_splits)) * latents1[0] + math.sqrt(i / num_splits) * latents2[0],
            math.sqrt(1 - (i / num_splits)) * latents1[1] + math.sqrt(i / num_splits) * latents2[1],
        ]

        recon = coupled_stablediffusion(
            unet=unet,
            vae=vae,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            prompt=caption,
            reverse=False,
            fixed_starting_latent=latents,
            run_baseline=run_baseline,
            use_double=use_double
        )
        recon = recon[0]
        axes[i].imshow(recon)
        axes[i].set_title(f"{i / num_splits:.2f}")
        if not os.path.isdir(f"logs/{caption.replace(' ', '_')}"):
            os.makedirs(f"logs/{caption.replace(' ', '_')}", exist_ok=True)
        recon.save(f"logs/{caption.replace(' ', '_')}/{i}.png")
    plt.tight_layout()
    plt.savefig(f"logs/{caption.replace(' ', '_')}.png")


    # 
    # ax0.imshow(im)
    # ax0.set_title("Original")
    # ax1.imshow(recon)
    # ax1.set_title("Recon")
    # plt.tight_layout()
    # plt.savefig(f"{save_dir}/{item['image_id']}_{item['id']}.png")


    # # im = load_im_into_format_from_path('https://raw.githubusercontent.com/salesforce/EDICT/main/experiment_images/church.jpg')
    # # load data
    # json_path = os.path.join(data_dir, "coco", "annotations", "captions_val2017.json")
    # with open(json_path, "r") as f:
    #     data = json.load(f)

    # annotation = data["annotations"]
    # annotation = annotation[get_rank()::get_world_size()]

    # for item in tqdm(annotation, total=len(annotation), disable=get_rank() != 0):
    #     im_path = os.path.join(data_dir, "coco", "val2017", f"{item['image_id']:012d}.jpg")
    #     im = load_im_into_format_from_path(im_path)
    #     prompt = item["caption"]

    #     latents = coupled_stablediffusion(
    #         unet=unet,
    #         vae=vae,
    #         clip=clip,
    #         clip_tokenizer=clip_tokenizer,
    #         init_image=im,
    #         prompt=prompt,
    #         run_baseline=run_baseline,
    #         reverse=True,
    #         use_double=use_double
    #     )

    #     if run_baseline:
    #         latents = latents[0]

    #     recon = coupled_stablediffusion(
    #         unet=unet,
    #         vae=vae,
    #         clip=clip,
    #         clip_tokenizer=clip_tokenizer,
    #         prompt=prompt,
    #         reverse=False,
    #         fixed_starting_latent=latents,
    #         run_baseline=run_baseline,
    #         use_double=use_double
    #     )
    #     recon = recon[0]

    #     fig, (ax0, ax1) = plt.subplots(1, 2)
    #     ax0.imshow(im)
    #     ax0.set_title("Original")
    #     ax1.imshow(recon)
    #     ax1.set_title("Recon")
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/{item['image_id']}_{item['id']}.png")


@cli.command()
@click.option('--use-double', type=bool, default=False)
@click.option('--init-image-strength', type=float, default=0.8)
@click.option('--seed', type=int, default=0)
def edict_demo(use_double=False, data_dir="../datasets", init_image_strength=0.8, seed=42):
    r"""
    """

    dist_init()

    save_dir = f"logs/results-seed{seed}" if not use_double else f"logs/results-seed{seed}_double"
    os.makedirs(save_dir, exist_ok=True)

    # set random seed
    SEED = seed + get_rank()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # load model
    auth_token = ""
    # Build our CLIP model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip, cache_dir="../.cache")
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16, cache_dir="../.cache")
    clip = clip_model.text_model

    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16, cache_dir="../../.cache")

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

    print0("Loaded all models")

    if True:

        im_path = "https://i.pinimg.com/originals/1d/fd/d0/1dfdd00f6e23a2ecb3eee13529e5db4b.jpg"
        input_text = "a truck parking under starry night"
        output_text = "an old ship parking under starry night"
        instruction = "change the truck into an old ship"
        out_path = os.path.join(save_dir, im_path.replace(data_dir, "").replace("/", "__"))
        main_worker(
            unet=unet,
            vae=vae,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            im_path=im_path,
            instruction=instruction,
            out_path=out_path,
            base_prompt=input_text,
            edit_prompt=output_text,
            init_image_strength=init_image_strength,
            use_double=use_double
        )



if __name__ == "__main__":
    r"""
    package requirements:
        diffusers==0.10.0
        transformers==4.32.1

    ```
        python run.py edict-demo
    ```
    """

    cli()