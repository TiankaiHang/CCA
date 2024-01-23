import os
import sys
import re

import torch

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from diffusers import AutoPipelineForInpainting
import shutil

# src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.generator import grounding_dino_inpainting_api
from src.generator import edict_api
from utils import prompts, concat_image_in_row
from src.aesthetics_score import get_aesthetic_score
from src.llava_cli import llava_caption_api
from src.generator import (
    instruct_diffusion_api,
    grounding_dino_api,
)


__all__ = [
    "Resize",
    "InstructDiffusion",
    "LLaVA",
    "AestheticScore",
    "ImageDifferenceLLaVA",
    "ImageCommonPointLLaVA",
    "EdictEditing",
    "EdictEditingP2P",
    "GroundingDINO_Inpainting",
    "Crop",
    "RGB2Gray",
    "GaussianBlur",
    "SDXLOutpainting",
    "RotateClockwise",
    "RotateCounterClockwise",
    "EnhanceColor",
    "BlurBackground",
    "AddWatermark",
    "AddLogo",
    "BlackWhiteBackground",
    "RemoveBackground",
    "FlipHorizontal",
    "GroundingDINO",
    "SDXLInpainting",
    "CenterCrop",
]


class Resize(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="resize-image",
             description="resize the image to the given resolution/long side. "
                         "Useful when you want to resize the image to make the long side to a specific number. ",
             cookbook="receives image_path and the resolution as input. " +
                      "The input to this tool should be a <-> separated string of three, " +
                      "representing the `input image path`, `save path` and the `resolution` (int value). " +
                      "no other special instructions are needed.")
    def inference(self, inputs):
        # image_path, resolution = inputs.split("<->")[0], ','.join(inputs.split(',')[1:])
        image_path, target_path, resolution = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2]
        # image_path = re.findall(r"image/.*\.(?:png|jpg|jpeg)", image_path)[0]
        # target_path = re.findall(r"image/.*\.(?:png|jpg|jpeg)", target_path)[0]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        # resolution = int(resolution)
        resolution = int(re.findall(r"\d+", resolution)[0])

        image = Image.open(image_path)

        width, height = image.size
        ratio = min(resolution / width, resolution / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        image = image.resize((width_new, height_new))

        # updated_image_path = get_new_image_name(image_path, func_name="resize")
        image.save(target_path)
        print(
            f"\nProcessed Resize, Input Image: {image_path}, Resolution: {resolution}, "
            f"Output Image: {target_path}")
        return target_path


class InstructDiffusion(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="instructdiffusion",
             description=f"Useful when you want to edit the image with a text instruction. "
                         f"Can be used for local editing like adding/removing/replacing objects, "
                         f"and global editing like changing the style of the image. ",
             #  f"This tool is usually used as the backup tool when the other tools fail to generate the desired image.",
             cookbook=f"The random seed is a int value, playing a crucial role in the diversity and variability of the generated images. Default is 42. It is encouraged to change the random seed. " + \
                      f"The text classifier-free guidance (`txt-cfg`) is a float value, strictly greater than 1.0 and less than 10.0, default is 7.5. " + \
                      f"The image classifier-free guidance (`img-cfg`) is a float value, strictly greater than 0.0 and less than 2.0, default is 1.25. " + \
                      f"A larger `txt-cfg` value results in an output image showing more editing effects, " + \
                      f"while a larger `img-cfg` value leads to an output image more similar to the input image." + \
                      f"`txt-cfg` increases or decreases in steps of 1.0." + \
                      f"The input to this tool should be a <-> separated string of six, " + \
                      f"representing `input image path`, `save path`, `random seed`, `text classifier-free guidance (txt-cfg)`, `image classifier-free guidance (img-cfg)`, and `text`. " + \
                      f"no other special instructions are needed.")
    def inference(self, inputs):
        # image_path, target_path, text = inputs.split("<->")[0], inputs.split("<->")[1], ','.join(inputs.split(',')[2:])
        image_path, target_path, seed, txt_cfg, img_cfg, text = inputs.split("<->")[0], inputs.split("<->")[1], inputs.split(
            "<->")[2], inputs.split("<->")[3], inputs.split("<->")[4], ','.join(inputs.split("<->")[5:])

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        txt_cfg = float(txt_cfg)
        img_cfg = float(img_cfg)

        seed = int(seed)

        # ckpt list
        # checkpoints/MagicBrush-epoch-52-step-4999.ckpt
        # checkpoints/instruct-pix2pix-00-22000.ckpt
        # checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt
        # checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt

        ckpt = os.environ.get(
            "CKPT", "checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt")
        # ckpt = os.environ.get("CKPT", "checkpoints/MagicBrush-epoch-52-step-4999.ckpt")

        image = instruct_diffusion_api(
            cfg_file="src/generator/instructdiffusion/configs/instruct_diffusion.yaml",
            # ckpt        = "checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt",
            ckpt=ckpt,
            resolution=512,
            input_img=image_path,
            edit=text,
            seed=seed,
            steps=50,
            cfg_text=txt_cfg,
            cfg_image=img_cfg,
            use_cache=False,
        )

        image.save(target_path)
        print(
            f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
            f"Output Image: {target_path}")
        return target_path


class LLaVA(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="llava-caption",
             description=f"desribe or answer the questions about the image. "
                         f"Useful when you want to know what is in the photo, "
                         f"and judge the existence of objects in the photo. ",
             cookbook=f"receives image_path and the instruction text as input. " +
                      f"The instruction text can be `Provide a one-sentence caption for the provided image.` " +
                      f"Or questions like `Is there a cat in the photo`, " +
                      f"`how many people are in the photo`, " +
                      f"The tool input should be a <-> separated string of two, " +
                      f"representing the `input image path` and the `text`. ")
    def inference(self, inputs):
        image_path, text = inputs.split(
            "<->")[0], ','.join(inputs.split("<->")[1:])
        # image_path = re.findall(r"image/.*\.(?:png|jpg|jpeg)", image_path)[0]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]

        image_caption = llava_caption_api(
            model_path="liuhaotian/llava-v1.5-7b",
            image_file=image_path,
            model_base=None,
            device=self.device,
            load_8bit=False,
            load_4bit=True,
            prompt=text,
            temperature=0.2,
            verbose=False,
            use_cache=False,
        )
        print(
            f"\nProcessed LLaVA, Input Image: {image_path}, Instruct Text: {text}, "
            f"Output Text: {image_caption}")
        return image_caption


class AestheticScore(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="aesthetic-score",
             description=f"get the aesthetic score of the image. "
                         f"Useful when you want to know how beautiful the photo is.",
             cookbook=f"receives image_path as input. " +
                      f"The input to this tool should be a string, representing the input image path. " +
                      f"no other special instructions are needed.")
    def inference(self, image_path):
        # image_path = re.findall(r"image/.*\.(?:png|jpg|jpeg)", image_path)[0]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]

        score = get_aesthetic_score(image_path)
        score = round(score, 3)
        print(
            f"\nProcessed AestheticScore, Input Image: {image_path}, Output Score: {score}")
        return score


class ImageDifferenceLLaVA(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="image-difference-llava",
             description=f"desribe the difference between two images. "
                         f"Useful when you want to know what is the difference between two photos.",
             cookbook=f"receives image1 path and image2 path as input. " +
             f"The input to this tool should be a <-> separated string of two, " +
             f"representing the `input image1 path` and the `image2 path`. ")
    def inference(self, inputs):
        # image_path, text = inputs.split("<->")[0], ','.join(inputs.split(',')[1:])
        image_path1, image_path2 = inputs.split(
            "<->")[0], inputs.split("<->")[1]
        try:
            # image_path1 = re.findall(r"image/.*\.(?:png|jpg|jpeg)", image_path1)[0]
            # image_path2 = re.findall(r"image/.*\.(?:png|jpg|jpeg)", image_path2)[0]

            pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
            image_path1 = re.findall(pattern, image_path1)[0]
            image_path2 = re.findall(pattern, image_path2)[0]

            concat_image_in_row(
                image_path1, image_path2).save("image/concat.png")
        except FileNotFoundError:
            print(
                f"FileNotFoundError: {image_path1} or {image_path2} not found")
            return f"FileNotFoundError: {image_path1} or {image_path2} not found"

        text = "The left and right part are two different images. What are the differences between them?"
        image_caption = llava_caption_api(
            model_path="liuhaotian/llava-v1.5-7b",
            image_file="image/concat.png",
            model_base=None,
            device=self.device,
            load_8bit=False,
            load_4bit=True,
            prompt=text,
            temperature=0.2,
            verbose=False,
        )
        print(
            f"\nProcessed LLaVA, Input Image: {image_path1}, {image_path2}, Instruct Text: {text}"
            f"Output Text: {image_caption}")
        return image_caption


class ImageCommonPointLLaVA(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="image-common-llava",
             description=f"desribe the common points between two images. "
                         f"Useful when you want to know what are shared characteristics between two photos.",
             cookbook=f"receives image1 path and image2 path as input. " +
                      f"The input to this tool should be a <-> separated string of two, " +
                      f"representing the `input image1 path` and the `image2 path`. ")
    def inference(self, inputs):
        # image_path, text = inputs.split("<->")[0], ','.join(inputs.split(',')[1:])
        image_path1, image_path2 = inputs.split(
            "<->")[0], inputs.split("<->")[1]
        try:
            # image_path1 = re.findall(r"image/.*\.(?:png|jpg|jpeg)", image_path1)[0]
            # image_path2 = re.findall(r"image/.*\.(?:png|jpg|jpeg)", image_path2)[0]

            pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
            image_path1 = re.findall(pattern, image_path1)[0]
            image_path2 = re.findall(pattern, image_path2)[0]

            concat_image_in_row(
                image_path1, image_path2).save("image/concat.png")
        except FileNotFoundError:
            print(
                f"FileNotFoundError: {image_path1} or {image_path2} not found")
            return f"FileNotFoundError: {image_path1} or {image_path2} not found"

        text = "The left and right part are two different images. What are the differences between them?"
        image_caption = llava_caption_api(
            model_path="liuhaotian/llava-v1.5-7b",
            image_file="image/concat.png",
            model_base=None,
            device=self.device,
            load_8bit=False,
            load_4bit=True,
            prompt=text,
            temperature=0.2,
            verbose=False,
        )
        print(
            f"\nProcessed LLaVA, Input Image: {image_path1}, {image_path2}, Instruct Text: {text}"
            f"Output Text: {image_caption}")
        return image_caption


# ------------------------------------------------------------------------------
# Edict Editing
# ------------------------------------------------------------------------------


class EdictEditing(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="edict-editing", description=f"Useful when you want to edit the image with a text instruction. " +
             f"Can be used for object replacement.", cookbook="receives `input image path`, `save path`, `base prompt`, and `edit prompt` as input. " +
             "The base prompt is the text describing the input image. " +
             "The edit prompt is the text describing the edited image. " +
             "The base prompt and edit prompt should be descriptive enough. " +
             "Input image path and save path are the path to the input image and the path to save the output image. " +
             "For example, if I want to add a hat to the image of a cute dog, " +
             "the base prompt could be `a cute dog running on the ground`, and the edit prompt could be `a cute dog with a hat running on the ground`. " +
             "or if i want to replace the dog with a cat, the edit prompt could be `a cute cat running on the ground`. " +
             "Then the input to this tool should be a <-> separated string of four, " +
             "representing the `input image path`, the `save path`, the `base prompt`, and the `edit prompt`. " +
             "no other special instructions are needed.")
    def inference(self, inputs):
        sep_parts = inputs.split("<->")
        image_path, target_path, base_prompt, edit_prompt = sep_parts[
            0], sep_parts[1], sep_parts[2], sep_parts[3]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'

        # parse the image path
        # image_path = re.findall(pattern, image_path)[0]
        # target_path = re.findall(pattern, target_path)[0]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = edict_api(
            img_path=image_path,
            base_prompt=base_prompt,
            edit_prompt=edit_prompt,
            resolution=512,
            use_p2p=False,
        )

        image.save(target_path)
        print(
            f"\nProcessed EdictEditing Input Image: {image_path}, Base Prompt: {base_prompt}, Edit Prompt: {edit_prompt}, "
            f"Output Image: {target_path}")
        return target_path


class EdictEditingP2P(object):
    r"""
    with prompt-to-prompt version.
    ensure the background not changed.
    """

    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="edict-editing",
             description=f"Useful when you want to edit the image with a text instruction, especially for object replacement. " +
                         f"Recieves the image and text as input and outputs the edited image.",
             cookbook="receives `input image path`, `save path`, `base prompt`, and `edit prompt` as input. " +
                      "The base prompt is the text describing the input image. " +
                      "The edit prompt is the text describing the edited image. " +
                      "The base prompt and edit prompt should be descriptive enough. " +
                      "Input image path and save path are the path to the input image and the path to save the output image. " +
                      "For example, if I want to add a hat to the image of a cute dog, " +
                      "the base prompt could be `a cute dog running on the ground`, and the edit prompt could be `a cute dog with a hat running on the ground`. " +
                      "or if i want to replace the dog with a cat, the edit prompt could be `a cute cat running on the ground`. " +
                      "Then the input to this tool should be a <-> separated string of four, " +
                      "representing the `input image path`, the `save path`, the `base prompt`, and the `edit prompt`. " +
                      "no other special instructions are needed.")
    def inference(self, inputs):
        sep_parts = inputs.split("<->")
        image_path, target_path, base_prompt, edit_prompt = sep_parts[
            0], sep_parts[1], sep_parts[2], sep_parts[3]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'

        # parse the image path
        # image_path = re.findall(pattern, image_path)[0]
        # target_path = re.findall(pattern, target_path)[0]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = edict_api(
            img_path=image_path,
            base_prompt=base_prompt,
            edit_prompt=edit_prompt,
            resolution=512,
            use_p2p=True,
        )

        image.save(target_path)
        print(
            f"\nProcessed EdictEditing Input Image: {image_path}, Base Prompt: {base_prompt}, Edit Prompt: {edit_prompt}, "
            f"Output Image: {target_path}")
        return target_path


# ------------------------------------------------------------------------------
# Gounding + Edit
# ------------------------------------------------------------------------------


class GroundingDINO_Inpainting(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="grounding-dino-inpainting",
             description=f"Useful when you want to edit the image with text guidance. " +
                         f"Works well for object replacement. It detects the specific object in the image " +
                         f"using some prompt and inpaints the area with target prompt.",
             cookbook=f"receives `input image path`, `save path`, `detect prompt`, and `inpaint prompt` as input. " +
                      f"The `detect prompt` is the text describing the object to be detected. " +
                      f"The `inpaint prompt` is the text describing the area to be inpainted and should be specific. " +
                      f"For example, if I want to change the black car in the road to a cute dog, " +
                      f"the `detect prompt` could be 'black car', and the `inpaint prompt` could be 'a cute dog in the road'. " +
                      f"Or if I want to remove the balloon in the sky, the `detect prompt` could be `balloon`, and the `inpaint prompt` could be `sky`. " +
                      f"The input to this tool should be a <-> separated string of four, " +
                      f"representing the `input image path`, the `save path`, the `detect prompt`, and the `inpaint prompt`. ")
    def inference(self, inputs):
        sep_parts = inputs.split("<->")
        image_path, target_path, detect_prompt, inpaint_prompt = sep_parts[
            0], sep_parts[1], sep_parts[2], sep_parts[3]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        out_image, mask = grounding_dino_inpainting_api(
            # image_path=inputs.split("<->")[0],
            # text=inputs.split("<->")[1],
            # save_path=inputs.split("<->")[2],
            device=self.device,
            sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
            image_path=image_path,
            inpaint_mode="merge",
            inpaint_prompt=inpaint_prompt,
            box_threshold=0.3,
            text_threshold=0.25,
            config_file="src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounded_checkpoint="checkpoints/groundingdino_swint_ogc.pth",
            det_prompt=detect_prompt,
            resolution=512,
        )

        if out_image is None:
            print(f"GroundingDINO_Inpainting failed, no object detected. Skip ...")
            shutil.copy(image_path, target_path)
        else:
            mask.save(target_path.replace(".png", "_mask.png"))
            out_image.save(target_path)

        print(
            f"\nProcessed GroundingDINO_Inpainting Input Image: {image_path}, Detect Prompt: {detect_prompt}, Inpaint Prompt: {inpaint_prompt}, "
            f"Output Image: {target_path}")
        return target_path


class GroundingDINO(object):
    r"""
    detect and get the mask
    """

    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="grounding-dino",
             description=f"detect specific object and get the mask instead of natural image given the prompt. " +
                         f"Recieves the image and text as input and outputs the mask. Mask is NOT the image " +
                         f"and only contains semantic information.",
             cookbook=f"receives `input image path`, `detect prompt` as input. " +
                      f"The `detect prompt` is the text describing the object to be detected. " +
                      f"For example, if I want to get the mask of a black car in the image, " +
                      f"the `detect prompt` could be 'black car'. " +
                      f"The input to this tool should be a <-> separated string of two, " +
                      f"representing the `input image path`, and the `detect prompt`. ")
    def inference(self, inputs):
        sep_parts = inputs.split("<->")
        # image_path, target_path, detect_prompt = sep_parts[0], sep_parts[1], sep_parts[2]
        image_path, detect_prompt = sep_parts[0], sep_parts[1]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        # target_path = re.findall(pattern, target_path)[0]
        suffix = image_path.split(".")[-1]
        target_path = image_path.replace(f".{suffix}", f"_mask.{suffix}")

        _, mask = grounding_dino_api(
            device=self.device,
            sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
            image_path=image_path,
            inpaint_mode="",
            box_threshold=0.3,
            text_threshold=0.25,
            config_file="src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounded_checkpoint="checkpoints/groundingdino_swint_ogc.pth",
            det_prompt=detect_prompt,
        )

        # save mask
        mask.save(target_path)

        print(
            f"\nProcessed GroundingDINO, Input Image: {image_path}, Detect Prompt: {detect_prompt}, "
            f"Output Image: {target_path}")

        return target_path


class SDXLInpainting(object):
    r"""
    Inpaint the image given the mask
    """

    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="sdxl-inpainting",
             description=f"This tool allows you to replace a specific area in an image (defined by a mask) " +
                         f"with new content based on a provided description. " +
                         f"Recieves the image, mask and text as input and outputs the inpainted image.",
             cookbook=f"receives `input image path`, `save path`, `mask path`, and `inpaint prompt` as input. " +
                      f"The `inpaint prompt` is the text describing the area to be inpainted and should be specific. " +
                      f"The mask path is the path to the mask image, it shoud contain `_mask`. " +
                      f"For example, if I want to inpaint the masking area to be a cute dog, " +
                      f"the `inpaint prompt` could be 'a cute dog'. " +
                      f"The input to this tool should be a <-> separated string of four, " +
                      f"representing the `input image path`, the `save path`, the `mask path`, and the `inpaint prompt`. ")
    def inference(self, inputs):
        sep_parts = inputs.split("<->")
        image_path, target_path, mask_path, inpaint_prompt = sep_parts[
            0], sep_parts[1], sep_parts[2], sep_parts[3]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]
        mask_path = re.findall(pattern, mask_path)[0]

        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16, variant="fp16").to("cuda")

        image_pil = Image.open(image_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("RGB")
        mask_pil.resize(image_pil.size)

        image = pipe(
            prompt=inpaint_prompt,
            image=image_pil,
            mask_image=mask_pil,
            guidance_scale=8.0,
            num_inference_steps=20,  # steps between 15 and 30 work well for us
            strength=0.98,  # make sure to use `strength` below 1.0
            # generator=generator,
        ).images[0]

        image = image.resize(image_pil.size)
        image.save(target_path)

        print(
            f"\nProcessed SDXLInpainting, Input Image: {image_path}, Mask Path: {mask_path}, Inpaint Prompt: {inpaint_prompt}, "
            f"Output Image: {target_path}")

        return target_path


# ------------------------------------------------------------------------------
# add more tools 11/16
# ------------------------------------------------------------------------------
class Crop(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="crop",
             description=f"crop the image to the given resolution/long side. "
             f"Useful when you want to crop the image to make the long side to a specific number. ",
             cookbook=f"receives input image path, save path, and the resolution as input. " +
                         f"The input to this tool should be a <-> separated string of three, " +
                         f"representing the `input image path`, `save path` and the `resolution` (int value). " +
                         f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path, resolution = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]
        resolution = int(re.findall(r"\d+", resolution)[0])

        image = Image.open(image_path)
        width, height = image.size

        min_len = min(width, height)
        if min_len < resolution:
            # resize first
            ratio = min(resolution / width, resolution / height)
            width_new, height_new = (
                round(width * ratio), round(height * ratio))
            image = image.resize((width_new, height_new))

        # crop the image
        width, height = image.size
        left = (width - resolution) / 2
        top = (height - resolution) / 2
        right = (width + resolution) / 2
        bottom = (height + resolution) / 2
        image = image.crop((left, top, right, bottom))

        image.save(target_path)
        print(
            f"\nProcessed Crop, Input Image: {image_path}, Resolution: {resolution}, "
            f"Output Image: {target_path}")
        return target_path


class CenterCrop(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="center-crop",
             description=f"crop the image at the center",
             cookbook=f"receives input image path, save path as input. " +
                         f"The input to this tool should be a <-> separated string of two, " +
                         f"representing the `input image path`, `save path`. " +
                         f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path = inputs.split(
            "<->")[0], inputs.split("<->")[1]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = Image.open(image_path)

        # center crop the image
        width, height = image.size
        resolution = min(width, height)
        left = int((width - resolution) / 2)
        top = int((height - resolution) / 2)
        right = int((width + resolution) / 2)
        bottom = int((height + resolution) / 2)
        image = image.crop((left, top, right, bottom))

        image.save(target_path)
        print(f"\nProcessed Crop, Input Image: {image_path}, "
              f"Output Image: {target_path}")
        return target_path


class RGB2Gray(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="rgb2gray",
             description=f"convert the image from rgb to gray. ",
             cookbook=f"receives input image path, save path as input. " +
                      f"The input to this tool should be a <-> separated string of two, " +
                      f"representing the `input image path` and the `save path`. " +
                      f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path = inputs.split(
            "<->")[0], inputs.split("<->")[1]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = Image.open(image_path)
        image = image.convert('L')
        image.save(target_path)
        print(f"\nProcessed RGB2Gray, Input Image: {image_path}, "
              f"Output Image: {target_path}")
        return target_path


class GaussianBlur(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="gaussian-blur",
             description=f"blur the image with gaussian filter. ",
             cookbook=f"receives input image path, save path, and the kernel size as input. " +
                      f"The input to this tool should be a <-> separated string of three, " +
                      f"representing the `input image path`, `save path` and the `kernel size` (int value). " +
                      f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path, kernel_size = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]
        kernel_size = int(re.findall(r"\d+", kernel_size)[0])

        image = Image.open(image_path)
        image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        image.save(target_path)
        print(
            f"\nProcessed GaussianBlur, Input Image: {image_path}, Kernel Size: {kernel_size}, "
            f"Output Image: {target_path}")
        return target_path


class SDXLOutpainting(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="sdxl-outpainting",
             description=f"outpaint the image around the center. ",
             cookbook=f"receives input image path, save path, inpaint prompt, and the outpaint size as input. " +
             f"The input to this tool should be a <-> separated string of four, " +
             f"representing the `input image path`, `save path`, `inpaint prompt` and the `outpaint size` (int value). " +
             f"no other special instructions are needed.")
    def inference(self, inputs):

        def pad_reflection(im, padding):
            mode = "constant"
            constant_values = 255
            return np.pad(im, [(padding, padding), (padding, padding),
                          (0, 0)], mode=mode, constant_values=constant_values)

        image_path, target_path, inpaint_prompt, outpaint_size = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2], inputs.split("<->")[3]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        outpaint_size = int(re.findall(r"\d+", outpaint_size)[0])

        image = Image.open(image_path)
        # width, height = image.size
        new_width = image.size[0] + outpaint_size * 2
        new_height = image.size[1] + outpaint_size * 2
        # new_image = Image.new("RGB", (new_width, new_height))
        # new_image.paste(image, (outpaint_size, outpaint_size))
        new_image = Image.fromarray(
            pad_reflection(
                np.array(image),
                outpaint_size))
        _w, _h = new_image.size

        mask_pil = Image.new("RGB", (new_width, new_height))
        mask_pil.paste((255, 255, 255), (0, 0, new_width, new_height))
        mask_pil.paste(
            (0,
             0,
             0),
            (outpaint_size,
             outpaint_size,
             outpaint_size +
             image.size[0],
             outpaint_size +
             image.size[1]))
        # mask_pil.paste((0, 0, 0), (0, 0, new_width, new_height))
        # mask_pil.paste((255, 255, 255), (outpaint_size, outpaint_size, outpaint_size + image.size[0], outpaint_size + image.size[1]))

        # multiple of 64
        new_width = int(np.round(new_width / 64.0)) * 64
        new_height = int(np.round(new_height / 64.0)) * 64
        new_image = new_image.resize((new_width, new_height))
        mask_pil = mask_pil.resize((new_width, new_height))

        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16, variant="fp16").to("cuda")

        image = pipe(
            prompt=inpaint_prompt,
            image=new_image,
            mask_image=mask_pil,
            guidance_scale=8.0,
            num_inference_steps=20,  # steps between 15 and 30 work well for us
            strength=0.98,  # make sure to use `strength` below 1.0
            # generator=generator,
        ).images[0]

        image = image.resize((_w, _h))
        image.save(target_path)
        mask_pil.resize(
            (_w, _h)).save(
            target_path.replace(
                ".png", "_mask.png"))
        print(
            f"\nProcessed SDXLOutpainting, Input Image: {image_path}, Inpaint Prompt: {inpaint_prompt}, Outpaint Size: {outpaint_size}, "
            f"Output Image: {target_path}")

        return target_path


class RotateClockwise(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="rotate-clockwise",
             description=f"rotate the image clockwise. ",
             cookbook=f"receives input image path, save path  as input. " +
             f"The input to this tool should be a <-> separated string of two, " +
             f"representing the `input image path` and `save path` . " +
             f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path = inputs.split(
            "<->")[0], inputs.split("<->")[1]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = Image.open(image_path)
        image = image.rotate(-90, expand=True)
        image.save(target_path)
        print(f"\nProcessed RotateClockwise, Input Image: {image_path}, "
              f"Output Image: {target_path}")
        return target_path


class RotateCounterClockwise(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="rotate-counter-clockwise",
             description=f"rotate the image counterclockwise. ",
             cookbook=f"receives input image path, save path  as input. " +
             f"The input to this tool should be a <-> separated string of two, " +
             f"representing the `input image path` and `save path` . " +
             f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path = inputs.split(
            "<->")[0], inputs.split("<->")[1]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = Image.open(image_path)
        image = image.rotate(90, expand=True)
        image.save(target_path)
        print(
            f"\nProcessed RotateCounterClockwise, Input Image: {image_path}, "
            f"Output Image: {target_path}")
        return target_path


class FlipHorizontal(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="flip-horizontal",
             description=f"flip the image horizontally. ",
             cookbook=f"receives input image path, save path  as input. " +
             f"The input to this tool should be a <-> separated string of two, " +
             f"representing the `input image path` and `save path` . " +
             f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path = inputs.split(
            "<->")[0], inputs.split("<->")[1]

        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image.save(target_path)
        print(f"\nProcessed FlipHorizontal, Input Image: {image_path}, "
              f"Output Image: {target_path}")
        return target_path


# blur the background

class BlurBackground(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="blur-background",
             description=f"detect the object in the image and blur the background. ",
             cookbook=f"receives input image path, save path, the detect prompt,  blur size as input. " +
                      f"The default blur size is 5."
                      f"The input to this tool should be a <-> separated string of four, " +
                      f"representing the `input image path`, `save path`, the `detect prompt`. and the `blur size` (int value). " +
                      f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path, detect_prompt, blur_size = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2], inputs.split("<->")[3]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        try:
            blur_size = int(re.findall(r"\d+", blur_size)[0])
        except BaseException:
            blur_size = 5

        image = Image.open(image_path)
        # width, height = image.size

        image, mask = grounding_dino_api(
            device=self.device,
            sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
            image_path=image_path,
            inpaint_mode="",
            box_threshold=0.3,
            text_threshold=0.25,
            config_file="src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounded_checkpoint="checkpoints/groundingdino_swint_ogc.pth",
            det_prompt=detect_prompt,
        )

        # bulr the background except the mask area
        new_image = image.filter(ImageFilter.GaussianBlur(radius=blur_size))
        new_image.paste(image, (0, 0), mask=mask)
        new_image.save(target_path)

        print(
            f"\nProcessed BlurBackground, Input Image: {image_path}, Detect Prompt: {detect_prompt}, "
            f"Output Image: {target_path}")
        return target_path


class RemoveBackground(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="remove-background",
             description=f"detect the object in the image and remove the background. ",
             cookbook=f"receives input image path, save path, the detect prompt. " +
                      f"The input to this tool should be a <-> separated string of four, " +
                      f"representing the `input image path`, `save path`, the `detect prompt`. " +
                      f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path, detect_prompt = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = Image.open(image_path)
        # width, height = image.size

        image, mask = grounding_dino_api(
            device=self.device,
            sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
            image_path=image_path,
            inpaint_mode="",
            box_threshold=0.3,
            text_threshold=0.25,
            config_file="src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounded_checkpoint="checkpoints/groundingdino_swint_ogc.pth",
            det_prompt=detect_prompt,
        )

        # set the background to white except the mask area
        new_image = Image.new("RGB", image.size, (255, 255, 255))
        new_image.paste(image, (0, 0), mask=mask)
        new_image.save(target_path)

        print(
            f"\nProcessed RemoveBackground, Input Image: {image_path}, Detect Prompt: {detect_prompt}, "
            f"Output Image: {target_path}")
        return target_path


class BlackWhiteBackground(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="blackwhite-background",
             description=f"detect the object in the image and make the background black and white. ",
             cookbook=f"receives input image path, save path, the detect prompt. " +
                      f"The input to this tool should be a <-> separated string of four, " +
                      f"representing the `input image path`, `save path`, the `detect prompt`. " +
                      f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path, detect_prompt = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        image = Image.open(image_path)
        # width, height = image.size

        image, mask = grounding_dino_api(
            device=self.device,
            sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
            image_path=image_path,
            inpaint_mode="",
            box_threshold=0.3,
            text_threshold=0.25,
            config_file="src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounded_checkpoint="checkpoints/groundingdino_swint_ogc.pth",
            det_prompt=detect_prompt,
        )

        # set the raw background to black and white except the mask area
        new_image = np.array(image)
        background_blackwhite = np.array(image.convert("L"))
        background_blackwhite = background_blackwhite[..., None]
        mask = np.array(mask)
        new_image = new_image * mask[..., None] + \
            background_blackwhite * (1 - mask[..., None])
        new_image = Image.fromarray(new_image.astype(np.uint8))
        new_image.save(target_path)

        print(
            f"\nProcessed BlackWhiteBackground, Input Image: {image_path}, Detect Prompt: {detect_prompt}, "
            f"Output Image: {target_path}")
        return target_path


class AddLogo(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="add-logo",
             description=f"add logo to specified place. ",
             cookbook=f"receives input image path, save path, the detect prompt, and the path of the logo. " +
                      f"Only support OpenAI logo now. It is at the path `logos/openai.png`."
                      f"The input to this tool should be a <-> separated string of four, " +
                      f"representing the `input image path`, `save path`, the `detect prompt`, and the `logo path`. " +
                      f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path, detect_prompt, logo_path = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2], inputs.split("<->")[3]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]
        logo_path = re.findall(pattern, logo_path)[0]

        image = Image.open(image_path)
        logo = Image.open(logo_path)

        image, mask, bboxes = grounding_dino_api(
            device=self.device,
            sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
            image_path=image_path,
            inpaint_mode="",
            box_threshold=0.3,
            text_threshold=0.25,
            config_file="src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounded_checkpoint="checkpoints/groundingdino_swint_ogc.pth",
            det_prompt=detect_prompt,
            return_boxes=True,
        )

        # get the center of the box
        bbox = bboxes[0]
        bbox = [int(x) for x in bbox]
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        target_logo_size = min(box_w, box_h) // 2

        logo = logo.resize((target_logo_size, target_logo_size))

        # paste logo to the mask area, the center of the logo is the center of
        # the box
        new_image = image.copy()
        new_image.paste(
            logo,
            (bbox[0] +
             box_w //
             2 -
             target_logo_size //
             2,
             bbox[1] +
                box_h //
                2 -
                target_logo_size //
                2),
            mask=logo)
        new_image.save(target_path)

        print(
            f"\nProcessed AddLogo, Input Image: {image_path}, Detect Prompt: {detect_prompt}, "
            f"Output Image: {target_path}")
        return target_path


# class AddWatermark(object):
#     def __init__(self, device="cuda:0"):
#         self.device = device

#     @prompts(name="add-watermark",
#              description=f"add watermark from specified path to the background. ",
#              cookbook=f"receives input image path, save path, the detect prompt, and the watermark image path. " + \
#                       f"Supported watermark images should be one of the following: " + \
#                       f"`logos/openai.png`, `logos/microsoft.png`. " + \
#                       f"The input to this tool should be a <-> separated string of four, " + \
#                       f"representing the `input image path`, `save path`, the `detect prompt`, and the `watermark image path`. " + \
#                       f"no other special instructions are needed.")
#     def inference(self, inputs):
#         image_path, target_path, detect_prompt, logo_path = inputs.split("<->")[0], inputs.split("<->")[1], inputs.split("<->")[2], inputs.split("<->")[3]
#         pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
#         image_path = re.findall(pattern, image_path)[0]
#         target_path = re.findall(pattern, target_path)[0]
#         logo_path = re.findall(pattern, logo_path)[0]

#         image = Image.open(image_path).convert("RGB")
#         logo = Image.open(logo_path).convert("RGB")

#         # logo_new = logo.resize((image.size[0] // 8, image.size[1] // 8))
#         w_logo, h_logo = logo.size
#         ratio_logo = w_logo / h_logo

#         new_w_logo = image.size[0] // 4
#         new_h_logo = int(new_w_logo / ratio_logo)

#         logo_new = logo.resize((new_w_logo, new_h_logo))

#         new_image = image.copy()
#         # blend the background with the logo
#         new_image = np.array(new_image)
#         logo_new = np.array(logo_new)
#         inds = logo_new.sum(axis=-1) > 200 * 3
#         # import pdb; pdb.set_trace()

#         h_logo, w_logo = logo_new.shape[:2]

#         # put logo in the bottom right corner
#         # new_image[-h_logo:, -w_logo:] = logo_new

#         _patch = new_image[-h_logo:, -w_logo:].copy()
#         logo_new[inds] = _patch[inds]
#         new_image[-h_logo:, -w_logo:] = logo_new

#         new_image = Image.fromarray(new_image.astype(np.uint8))
#         new_image.save(target_path)

#         print(f"\nProcessed AddLogo, Input Image: {image_path}, Detect Prompt: {detect_prompt}, "
#                 f"Output Image: {target_path}")
#         return target_path


class AddWatermark(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="add-watermark",
             description=f"add watermark from specified path to the image. ",
             cookbook=f"receives input image path, save path, and the watermark image path. " +
                      f"Supported watermark images should be one of the following: " +
                      f"`logos/openai.png`, `logos/microsoft.png`. " +
                      f"The input to this tool should be a <-> separated string of three, " +
                      f"representing the `input image path`, `save path`, and the `watermark image path`. " +
                      f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path, watermark_image_path = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]
        watermark_image_path = re.findall(pattern, watermark_image_path)[0]

        image = Image.open(image_path).convert("RGB")
        watermark_image = Image.open(watermark_image_path).convert("RGB")

        # logo_new = logo.resize((image.size[0] // 8, image.size[1] // 8))
        transparency = 64  #
        # watermark_image.putalpha(transparency)

        # resize the watermark image
        w_watermark, h_watermark = watermark_image.size
        ratio_watermark = w_watermark / h_watermark
        watermark_image = watermark_image.resize(
            (image.size[0] // 4, int(image.size[0] // 4 / ratio_watermark)))

        watermark_position = (
            image.width - watermark_image.width,
            image.height - watermark_image.height)

        watermark_image_grey = watermark_image.convert("L")
        watermark_mask = Image.eval(
            watermark_image_grey,
            lambda x: 0 if x > 240 else transparency)
        watermark_image.putalpha(watermark_mask)

        image.paste(watermark_image, watermark_position, watermark_image)
        image.save(target_path)

        print(f"\nProcessed AddLogo, Input Image: {image_path}, "
              f"Output Image: {target_path}")
        return target_path


class EnhanceColor(object):
    def __init__(self, device="cuda:0"):
        self.device = device

    @prompts(name="enhance-color",
             description=f"enhance the color of the image. ",
             cookbook=f"receives input image path, save path, and the enhance factor as input. " +
                         f"The enhance factor is a float value, default is 1.5. "
                         f"The input to this tool should be a <-> separated string of three, " +
                         f"representing the `input image path`, `save path`, and the `enhance factor` (float value). " +
                         f"no other special instructions are needed.")
    def inference(self, inputs):
        image_path, target_path, enhance_factor = inputs.split(
            "<->")[0], inputs.split("<->")[1], inputs.split("<->")[2]
        pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'
        image_path = re.findall(pattern, image_path)[0]
        target_path = re.findall(pattern, target_path)[0]

        try:
            # re to find the float value
            enhance_factor = float(re.findall(r"\d+\.\d+", enhance_factor)[0])
        except BaseException:
            enhance_factor = 1.5

        image = Image.open(image_path)
        image = ImageEnhance.Color(image).enhance(enhance_factor)
        image.save(target_path)
        print(
            f"\nProcessed EnhanceColor, Input Image: {image_path}, Enhance Factor: {enhance_factor}, "
            f"Output Image: {target_path}")
        return target_path


# ------------------------------------------------------------------------------
# debug
# ------------------------------------------------------------------------------
def debug():

    GroundingDINO().inference(
        "results/sdxl_turbo.png <-> cat",
    )

    # edict_api(
    #     img_path="/scratch/t-thang/code_base/editing-agents/logs/baselines/church-instruct-pix2pix-00-22000.ckpt.png",
    #     base_prompt="a cute dog",
    #     edit_prompt="a cute dog with a hat",
    # )

    # addlogo_tool = AddLogo()
    # addlogo_tool.inference("datasets/23/003.png <-> outputs/x_addlogo.png <-> hand <-> logos/openai.png")

    # addwatermark_tool = AddWatermark()
    # addwatermark_tool.inference("datasets/23/004.jpeg <-> outputs/x_addwatermark.png <-> the man <-> logos/microsoft.jpg")

    # blackwhitebackground_tool = BlackWhiteBackground()
    # blackwhitebackground_tool.inference("datasets/23/004.jpeg <-> outputs/x_bw.png <-> the man <-> logos/microsoft.png")

    # flip_tool = FlipHorizontal()
    # flip_tool.inference("datasets/23/004.jpeg <-> outputs/x_flip.png")

    # edict_tool = EdictEditing()
    # edict_tool.inference("outputs/imagenet_dog_1.jpeg <-> image1/1-edict_wo_p2p.png <-> A dog <-> A dog and a rainbow")
    # edict_tool.inference("outputs/imagenet_dog_1.jpeg <-> image1/2-edict_wo_p2p.png <-> A dog <-> A white cat")

    # ground_inpaint_tool = GroundingDINO_Inpainting()
    # ground_inpaint_tool.inference("outputs/debug.png <-> image/1-ground-inpaint.png <-> black car <-> A cute dog")

    # crop_tool = Crop()
    # crop_tool.inference("outputs/x.png <-> outputs/x_cropped.png <-> 128")

    # rgb2gray_tool = RGB2Gray()
    # rgb2gray_tool.inference("outputs/x.png <-> outputs/x_gray.png")

    # gaussian_blur_tool = GaussianBlur()
    # gaussian_blur_tool.inference("outputs/x.png <-> outputs/x_blur.png <-> 3")

    # sdxl_outpaint_tool = SDXLOutpainting()
    # sdxl_outpaint_tool.inference("outputs/x.png <-> outputs/x_outpaint.png <-> wooden frame edge  <-> 200")

    # rotate_clockwise_tool = RotateClockwise()
    # rotate_clockwise_tool.inference("outputs/x.png <-> outputs/x_rotate_clockwise.png")

    # blur_background_tool = BlurBackground()
    # blur_background_tool.inference("outputs/x.png <-> outputs/x_blur_background.png <-> bench <-> 5")

    # enhance_color_tool = EnhhangeColor()
    # enhance_color_tool.inference("outputs/x.png <-> outputs/x_enhance_color.png <-> 1.5")

    # [
    #     # InstructDiffusion().inference("datasets/cyberbunk.jpg <-> image1/cyberbunk.png <-> 0 <-> 7.5 <-> 1.5 <-> transform the style of the image to be sunflowers"),
    #     InstructDiffusion().inference("datasets/cyberbunk.jpg <-> image1/cyberbunk.png <-> 0 <-> 9 <-> 1.5 <-> transform the style of the image to be van gogh Cottages"),
    #     FlipHorizontal().inference("image1/cyberbunk.png <-> image1/cyberbunk_flip.png"),
    # ]

    # "edict-editing @@ image/agent2_1.png <-> image/agent2_2.png <-> an image with cherry blossoms replacing the snowflakes <-> add a snowman in the foreground of the image with cherry blossoms"

    # [
    #     EdictEditingP2P().inference("logs/dec10-v9t0.8-000005679-2023-12-12-00-36-01/image/agent2_1.png <-> logs/dec10-v9t0.8-000005679-2023-12-12-00-36-01/image/agent2_2.png <-> an image with cherry blossoms replacing the snowflakes <-> add a snowman in the foreground of the image with cherry blossoms"),
    # ]

    # AddWatermark().inference("resized_image.png <-> results/demo3_1.png <-> logos/microsoft.png")
    # CenterCrop().inference("resized_image.png <-> image1/demo3_1.png")

    # ------------------------------------------------------ #
    # final response - 1
    # cmds = [
    # Resize().inference("image1/demo3_0.png <-> image1/demo3_1.png <-> 640"),
    # EdictEditing().inference("image1/demo3_1.png <-> image1/demo3_2.png <-> clouds <-> rainbow"),
    # InstructDiffusion().inference("image1/demo3_2.png <-> image1/demo3_3.png <-> 13 <-> 5.0 <-> 1.5 <-> replace the sorghum with a field of lavender"),
    # EnhhangeColor().inference("image1/demo3_3.png <-> image1/demo3_4.png <-> 1.5"),
    # ]

    # final response - 1
    # prefix = "image1/f1_"
    # cmds = [
    #     Resize().inference(f"image1/demo3_0.png <-> {prefix}1.png <-> 640"),
    #     EdictEditing().inference(f"{prefix}1.png <-> {prefix}2.png <-> clouds <-> rainbow"),
    #     InstructDiffusion().inference(f"{prefix}2.png <-> {prefix}3.png <-> 13 <-> 3.5 <-> 1.5 <-> replace the sorghum with a field of lavender"),
    #     EnhhangeColor().inference(f"{prefix}3.png <-> {prefix}4.png <-> 1.5"),
    # ]

    # ------------------------------------------------------ #
    # intial plan 1
    # prefix = "image1/init1_"
    # cmds = [
    #     Resize().inference(f"image1/demo3_0.png <-> {prefix}1.png <-> 640"),
    #     InstructDiffusion().inference(f"{prefix}1.png <-> {prefix}2.png <-> 13 <-> 3.0 <-> 1.5 <-> add rainbow in the sky"),
    #     InstructDiffusion().inference(f"{prefix}2.png <-> {prefix}3.png <-> 13 <-> 3.5 <-> 1.5 <-> replace the sorghum with a field of lavender"),
    #     EnhhangeColor().inference(f"{prefix}3.png <-> {prefix}4.png <-> 1.2"),
    # ]

    # ------------------------------------------------------ #
    # intial plan 2
    # prefix = "image1/init2_"
    # cmds = [
    # Resize().inference(f"image1/demo3_0.png <-> {prefix}1.png <-> 640"),
    # EdictEditing().inference(f"{prefix}1.png <-> {prefix}2.png <-> clouds <-> rainbow"),
    # InstructDiffusion().inference(f"{prefix}2.png <-> {prefix}3.png <-> 13 <-> 3.0 <-> 1.5 <-> replace the sorghum with a field of lavender"),
    # EnhhangeColor().inference(f"{prefix}3.png <-> {prefix}4.png <-> 1.2"),
    # ]

    # cmds = [
    #     Resize().inference("image1/demo3_0.png <-> {prefix}_1.png <-> 640"),
    #     EdictEditing().inference(f"{prefix}_2.png <-> {prefix}_3.png <-> clouds <-> rainbow"),
    #     EdictEditing().inference(f"{prefix}_1.png <-> {prefix}_2.png <-> 42 <-> 5.0 <-> 1.5 <-> replace the sorghum with a field of sunflowers"),
    #     EnhhangeColor().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 1.2"),
    # ]

    # import random
    # import torch
    # import numpy as np

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    # Initial Plan - 1
    # prefix = "image1/init1_"
    # cmds = [
    # Resize().inference(f"image1/demo3_0.png <-> {prefix}_1.png <-> 640"),
    # InstructDiffusion().inference(f"{prefix}_1.png <-> {prefix}_2.png <-> 13 <-> 3.0 <-> 1.5 <-> add a rainbow in the sky"),
    # GroundingDINO_Inpainting().inference(f"{prefix}_2.png <-> {prefix}_3.png <-> sorghum <-> a field of sunflowers"),
    # EnhhangeColor().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 1.2"),
    # ]

    # ## Initial Plan - 2
    # prefix = "image1/init2_"
    # cmds = [
    #     Resize().inference(f"image1/demo3_0.png <-> {prefix}_1.png <-> 640"),
    #     EdictEditing().inference(f"{prefix}_1.png <-> {prefix}_2.png <-> clouds <-> rainbow"),
    #     GroundingDINO_Inpainting().inference(f"{prefix}_2.png <-> {prefix}_3.png <-> sorghum <-> a field of sunflowers"),
    #     # EnhhangeColor().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 1.2"),
    #     InstructDiffusion().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 42 <-> 3.0 <-> 1.5 <-> enhance the color of the image"),
    # ]

    # prefix = "image1/final1_"
    # cmds = [
    #     Resize().inference(f"image1/demo3_0.png <-> {prefix}_1.png <-> 640"),
    #     EdictEditing().inference(f"{prefix}_1.png <-> {prefix}_2.png <-> clouds <-> rainbow"),
    #     GroundingDINO_Inpainting().inference(f"{prefix}_2.png <-> {prefix}_3.png <-> sorghum <-> a field of sunflowers"),
    #     EnhhangeColor().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 2.0"),
    # ]

    # prefix = "image1/final2_"
    # cmds = [
    #     Resize().inference(f"image1/demo3_0.png <-> {prefix}_1.png <-> 640"),
    #     EdictEditing().inference(f"{prefix}_1.png <-> {prefix}_2.png <-> clouds <-> rainbow in sky"),
    #     GroundingDINO_Inpainting().inference(f"{prefix}_2.png <-> {prefix}_3.png <-> field of sorghum <-> a field of sunflowers"),
    #     # EnhhangeColor().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 1.2"),
    #     EnhhangeColor().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 1.5"),
    # ]

    # Final Plan - 1
    # prefix = "image1/final1_"
    # cmds = [
    #     Resize().inference("image1/demo3_0.png <-> {prefix}_1.png <-> 640"),
    #     EdictEditing().inference(f"{prefix}_2.png <-> {prefix}_3.png <-> clouds <-> rainbow"),
    #     EdictEditing().inference(f"{prefix}_1.png <-> {prefix}_2.png <-> 42 <-> 5.0 <-> 1.5 <-> replace the sorghum with a field of sunflowers"),
    #     EnhhangeColor().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 1.2"),
    # ]

    # ## Final Plan - 2
    # prefix = "image1/final2_"
    # cmds = [
    #     Resize().inference("image1/demo3_0.png <-> {prefix}_1.png <-> 640"),
    #     EdictEditing().inference(f"{prefix}_2.png <-> {prefix}_3.png <-> clouds <-> rainbow"),
    #     EdictEditing().inference(f"{prefix}_1.png <-> {prefix}_2.png <-> 42 <-> 5.0 <-> 1.5 <-> replace the sorghum with a field of sunflowers"),
    #     EnhhangeColor().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 1.2"),
    # ]

    # prefix = "image1/111_"
    # cmds = [
    #     # Resize().inference(f"image1/demo3_0.png <-> {prefix}_1.png <-> 640"),
    #     # GroundingDINO_Inpainting().inference(f"{prefix}_1.png <-> {prefix}_2.png <-> sorghum <-> a field of sunflowers"),
    #     EdictEditing().inference(f"{prefix}_2.png <-> {prefix}_3.png <-> clouds in sky, sunflowers in the field <-> rainbow in sky, sunflowers in the field"),

    #     InstructDiffusion().inference(f"{prefix}_3.png <-> {prefix}_4.png <-> 31 <-> 6.0 <-> 1.5 <-> add a rustic wooden barn"),
    # ]


if __name__ == "__main__":
    debug()
