# ref: https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_inpainting_demo.py
import argparse
import os
import sys
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GD_DIR = os.path.join(BASE_DIR, "..", "..")
sys.path.append(GD_DIR)

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def enrose_image(mask_pil, kernel_size=13, iterations=10):
    mask_cv = np.array(mask_pil, dtype=np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    erosion = cv2.dilate(mask_cv, kernel, iterations=iterations)
    erosion[erosion > 0] = 255
    erosion_pil = Image.fromarray(erosion)

    return erosion_pil


def grounding_dino_inpainting_api(
    # output_dir = "outputs/",
    device              = "cuda:0",
    sam_checkpoint      = "checkpoints/sam_vit_h_4b8939.pth",
    image_path          = "outputs/debug.png",
    inpaint_mode        = "merge",
    inpaint_prompt      = "A cute dog",
    box_threshold       = 0.3,
    text_threshold      = 0.25,
    config_file         = "src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounded_checkpoint = "checkpoints/groundingdino_swint_ogc.pth",
    det_prompt          = "black car",
    resolution          = 512,
):
    # os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, _ = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    # import pdb; pdb.set_trace()
    if len(transformed_boxes) == 0:
        print("No box detected")
        return None

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # inpainting pipeline
    if inpaint_mode == 'merge':
        masks = torch.sum(masks, dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
    mask_pil = Image.fromarray(mask)
    image_pil = Image.fromarray(image)
    
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    # )
    # pipe = pipe.to("cuda")

    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
        torch_dtype=torch.float16, variant="fp16").to("cuda")

    # image_pil = image_pil.resize((resolution, resolution))
    # mask_pil = mask_pil.resize((resolution, resolution))

    # multiple of 64
    new_height = H - H % 64
    new_width = W - W % 64
    image_pil = image_pil.resize((new_width, new_height))
    mask_pil = mask_pil.resize((new_width, new_height))

    mask_pil = enrose_image(mask_pil, kernel_size=int(max(new_height, new_width) / 512.0 * 7), iterations=5)

    # prompt = "A sofa, high quality, detailed"
    # image = pipe(prompt=inpaint_prompt, image=image_pil, mask_image=mask_pil).images[0]
    image = pipe(
        prompt=inpaint_prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=8.0,
        num_inference_steps=20,  # steps between 15 and 30 work well for us
        strength=0.98,  # make sure to use `strength` below 1.0
        # generator=generator,
    ).images[0]
    
    image = image.resize(size)
    # image.save(os.path.join(output_dir, "grounded_sam_inpainting_output.jpg"))

    # save the mask
    mask = mask_pil.resize(size)
    # mask.save(os.path.join(output_dir, "mask.jpg"))

    return image, mask


def grounding_dino_api(
    # output_dir = "outputs/",
    device              = "cuda:0",
    sam_checkpoint      = "checkpoints/sam_vit_h_4b8939.pth",
    image_path          = "outputs/debug.png",
    inpaint_mode        = "merge",
    box_threshold       = 0.3,
    text_threshold      = 0.25,
    config_file         = "src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounded_checkpoint = "checkpoints/groundingdino_swint_ogc.pth",
    det_prompt          = "black car",
    resolution          = 512,
    return_boxes        = False,
):
    # os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, _ = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    # import pdb; pdb.set_trace()
    if len(transformed_boxes) == 0:
        print("No box detected")
        return None

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    # inpainting pipeline
    if inpaint_mode == 'merge':
        masks = torch.sum(masks, dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
    mask_pil = Image.fromarray(mask)
    image_pil = Image.fromarray(image)
    
    image = image_pil.resize(size)
    mask = mask_pil.resize(size)
    # mask.save(os.path.join(output_dir, "mask.jpg"))

    if return_boxes:
        return image, mask, boxes_filt

    return image, mask



def demo():

    # output_dir = "outputs/"
    # device = "cuda:0"
    # sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    # image_path = "outputs/debug.png"
    # inpaint_mode = "merge"
    # inpaint_prompt = "A cute dog"
    # box_threshold = 0.3
    # text_threshold = 0.25
    # config_file = "src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # grounded_checkpoint = "checkpoints/groundingdino_swint_ogc.pth"
    # det_prompt = "black car"

    # os.makedirs(output_dir, exist_ok=True)
    # # load image
    # image_pil, image = load_image(image_path)
    # # load model
    # model = load_model(config_file, grounded_checkpoint, device=device)

    # # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # # run grounding dino model
    # boxes_filt, pred_phrases = get_grounding_output(
    #     model, image, det_prompt, box_threshold, text_threshold, device=device
    # )

    # # initialize SAM
    # predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # predictor.set_image(image)

    # size = image_pil.size
    # H, W = size[1], size[0]
    # for i in range(boxes_filt.size(0)):
    #     boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
    #     boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
    #     boxes_filt[i][2:] += boxes_filt[i][:2]

    # boxes_filt = boxes_filt.cpu()
    # transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    # # import pdb; pdb.set_trace()
    # if len(transformed_boxes) == 0:
    #     print("No box detected")
    #     return None

    # masks, _, _ = predictor.predict_torch(
    #     point_coords = None,
    #     point_labels = None,
    #     boxes = transformed_boxes.to(device),
    #     multimask_output = False,
    # )

    # # masks: [1, 1, 512, 512]

    # # draw output image
    # # plt.figure(figsize=(10, 10))
    # # plt.imshow(image)
    # # for mask in masks:
    # #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # # for box, label in zip(boxes_filt, pred_phrases):
    # #     show_box(box.numpy(), plt.gca(), label)
    # # plt.axis('off')
    # # plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")

    # # inpainting pipeline
    # if inpaint_mode == 'merge':
    #     masks = torch.sum(masks, dim=0).unsqueeze(0)
    #     masks = torch.where(masks > 0, True, False)
    # mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
    # mask_pil = Image.fromarray(mask)
    # image_pil = Image.fromarray(image)
    
    # # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    # #     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    # # )
    # # pipe = pipe.to("cuda")

    # pipe = AutoPipelineForInpainting.from_pretrained(
    #     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
    #     torch_dtype=torch.float16, variant="fp16").to("cuda")

    # image_pil = image_pil.resize((512, 512))
    # mask_pil = mask_pil.resize((512, 512))
    # # prompt = "A sofa, high quality, detailed"
    # # image = pipe(prompt=inpaint_prompt, image=image_pil, mask_image=mask_pil).images[0]
    # image = pipe(
    #     prompt=inpaint_prompt,
    #     image=image_pil,
    #     mask_image=mask_pil,
    #     guidance_scale=8.0,
    #     num_inference_steps=20,  # steps between 15 and 30 work well for us
    #     strength=0.99,  # make sure to use `strength` below 1.0
    #     # generator=generator,
    # ).images[0]
        
    # image = image.resize(size)
    # image.save(os.path.join(output_dir, "grounded_sam_inpainting_output.jpg"))

    # # save the mask
    # mask = mask_pil.resize(size)
    # mask.save(os.path.join(output_dir, "mask.jpg"))

    image, mask = grounding_dino_inpainting_api(
        # output_dir = "outputs/",
        device              = "cuda:0",
        sam_checkpoint      = "checkpoints/sam_vit_h_4b8939.pth",
        image_path          = "outputs/debug.png",
        inpaint_mode        = "merge",
        inpaint_prompt      = "A cute corgi",
        box_threshold       = 0.3,
        text_threshold      = 0.25,
        config_file         = "src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounded_checkpoint = "checkpoints/groundingdino_swint_ogc.pth",
        det_prompt          = "a dog",
        resolution          = 512,
    )

    return image


if __name__ == "__main__":
    demo()