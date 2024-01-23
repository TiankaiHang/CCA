import os

import random
import numpy as np
import torch
from PIL import Image

import openai

import uuid
import tiktoken


def calculate_tokens(text, engine="gpt-35-turbo"):
    assert engine in ["gpt-35-turbo", "gpt-4"]
    enc = tiktoken.encoding_for_model(engine)
    tokens = enc.encode(text)
    
    return len(tokens)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# def prompts(name, description):
#     def decorator(func):
#         func.name = name
#         func.description = description
#         return func

#     return decorator

def prompts(name, description, cookbook):
    def decorator(func):
        func.name = name
        func.description = description
        func.cookbook = cookbook
        return func

    return decorator


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png'
    return os.path.join(head, new_file_name)


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


def load_image(img_path, resolution=768):
    image = Image.open(img_path)
    # image = image.resize((int(resolution), int(resolution)), Image.BICUBIC)

    width, height = image.size
    ratio = min(resolution / width, resolution / height)
    width_new, height_new = (round(width * ratio), round(height * ratio))
    width_new = int(np.round(width_new / 64.0)) * 64
    height_new = int(np.round(height_new / 64.0)) * 64
    image = image.resize((width_new, height_new))

    return image


def concat_image_in_row(img1, img2):
    img1 = load_image(img1)
    img2 = load_image(img2)

    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')
    
    if img1.size[1] != img2.size[1]:
        img1 = img1.resize((img2.size[0], img2.size[1]))
    img = np.concatenate([np.array(img1), np.array(img2)], axis=1)
    return Image.fromarray(img)

