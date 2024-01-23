from diffusers import DiffusionPipeline, DDIMScheduler
from transformers import CLIPTextModel
import torch, PIL, requests
from io import BytesIO


def center_crop_and_resize(im):

    width, height = im.size
    d = min(width, height)
    left = (width - d) / 2
    upper = (height - d) / 2
    right = (width + d) / 2
    lower = (height + d) / 2

    return im.crop((left, upper, right, lower)).resize((512, 512))


def load_image(img_path):
    if img_path.startswith("http"):
        response = requests.get(img_path)
        image = PIL.Image.open(BytesIO(response.content))
    else:
        image = PIL.Image.open(img_path)

    return image


def edict_api(img_path, base_prompt="A dog", edit_prompt="A golden retriever", use_double=False, resolution=-1):
    torch_dtype = torch.float16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # scheduler and text_encoder param values as in the paper
    scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            clip_sample=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path="openai/clip-vit-large-patch14",
        torch_dtype=torch_dtype,
    )

    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        custom_pipeline="edict_pipeline",
        revision="fp16",
        scheduler=scheduler,
        text_encoder=text_encoder,
        leapfrog_steps=True,
        torch_dtype=torch_dtype,
    ).to(device)

    # download image
    image = load_image(img_path)

    # preprocess it
    # cropped_image = center_crop_and_resize(image)
    image = image.resize((512, 512))

    # run the pipeline
    result_image = pipeline(
        base_prompt=base_prompt, 
        target_prompt=edit_prompt, 
        image=image,
    )
    import pdb; pdb.set_trace()
    result_image[0].save("outputs/xx.png")


if __name__ == "__main__":
    edict_api("https://huggingface.co/datasets/Joqsan/images/resolve/main/imagenet_dog_1.jpeg")
