import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if isinstance(image_file, str):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
    elif isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    else:
        raise ValueError("image_file must be a path or PIL.Image")
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, args)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


model_cache = {}

# wrapped api
def llava_caption_api(
        model_path  = "liuhaotian/llava-v1.5-7b",
        image_file  = "https://llava-vl.github.io/static/images/view.jpg",
        model_base  = None,
        device      = "cuda",
        load_8bit   = False,
        load_4bit   = False,
        prompt      = "describe the image in detail.",
        temperature = 0.2,
        verbose     = False,
        use_cache   = True,
):
    # kwargs in parser
    kwargs = {
        "model_path": model_path,
        "model_base": model_base,
        "image_file": image_file,
        "device": device,
        "conv_mode": None,
        "temperature": temperature,
        "max_new_tokens": 512,
        "load_8bit": load_8bit,
        "load_4bit": load_4bit,
        "debug": False,
        "image_aspect_ratio": 'pad',
    }

    # to namespace
    args = argparse.Namespace(**kwargs)

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if model_name not in model_cache and use_cache is False:
        tokenizer, model, image_processor, context_len = \
            load_pretrained_model(
                args.model_path, args.model_base, model_name, 
                args.load_8bit, args.load_4bit, device=args.device)
        model_cache[model_name] = (tokenizer, model, image_processor, context_len)
    else:
        tokenizer, model, image_processor, context_len = model_cache[model_name]

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)

    # width, height = image.size
    # if max(width, height) > 768:
    #     resolution = 768
    #     ratio = min(resolution / width, resolution / height)
    #     width_new, height_new = (round(width * ratio), round(height * ratio))
    #     width_new = int(np.round(width_new / 64.0)) * 64
    #     height_new = int(np.round(height_new / 64.0)) * 64
    #     image = image.resize((width_new, height_new))
    
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, args)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    if True:
        inp = prompt
        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

    return outputs.strip()


if __name__ == "__main__":

    output = llava_caption_api(
        model_path  = "liuhaotian/llava-v1.5-7b",
        image_file  = "https://llava-vl.github.io/static/images/view.jpg",
        model_base  = None,
        device      = "cuda",
        load_8bit   = False,
        load_4bit   = True,
        prompt      = "describe the image in detail.",
    )

    print(f"-------------------\n{output}\n-------------------")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--max-new-tokens", type=int, default=512)
    # parser.add_argument("--load-8bit", action="store_true")
    # parser.add_argument("--load-4bit", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    # args = parser.parse_args()
    # main(args)