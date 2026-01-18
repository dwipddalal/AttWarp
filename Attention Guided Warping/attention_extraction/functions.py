import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from transformers.generation.stopping_criteria import MaxNewTokensCriteria

from PIL import Image
import requests
from io import BytesIO
import re

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def getmask(args):
    # Model
    disable_torch_init()

    tokenizer, model, image_processor, context_len = args.tokenizer, args.model, args.image_processor, args.context_len

    hl = args.hl

    hl.reinit()

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in args.model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in args.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in args.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if isinstance(args.image_file, str):
        image_files = image_parser(args)
        images = load_images(image_files)
    elif isinstance(args.image_file, Image.Image):
        images = [args.image_file]
    else:
        raise ValueError("image_file should be str or PIL.Image")
    
    images = [image.convert('RGB') if image.mode != 'RGB' else image for image in images]

    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # Find and set image token range for the hook logger
    # In LLaVA, IMAGE_TOKEN_INDEX (-200) in input_ids is replaced with 576 image tokens
    # Find where the image token placeholder is
    input_ids_list = input_ids[0].tolist()
    try:
        image_token_pos = input_ids_list.index(IMAGE_TOKEN_INDEX)
    except ValueError:
        # If not found, assume image tokens start after BOS token
        image_token_pos = 1

    # LLaVA-1.5 uses 24x24 = 576 image patches
    num_image_tokens = 576

    # Set the image token range in the hook logger
    # Note: During generation, input_ids will be expanded to include image tokens
    # The actual positions will be: image_token_pos to image_token_pos + num_image_tokens
    hl.set_image_token_range(image_token_pos, image_token_pos + num_image_tokens)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [
        KeywordsStoppingCriteria(keywords, tokenizer, input_ids),
        MaxNewTokensCriteria(input_ids.shape[1], args.max_new_tokens)
    ]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            output_attentions=True,  # Required for attention hook to work
            return_dict_in_generate=True,
        )

    # Handle both GenerateOutput (dict-like) and tensor returns
    if hasattr(output_ids, 'sequences'):
        output_sequences = output_ids.sequences
    else:
        output_sequences = output_ids

    attention_output = hl.finalize().view(24, 24)

    input_token_len = input_ids.shape[1]
    output_len = output_sequences.shape[1]

    # Handle case where output contains only new tokens vs full sequence
    if output_len >= input_token_len:
        # Output includes input tokens - extract only generated part
        n_diff_input_output = (input_ids != output_sequences[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        generated_ids = output_sequences[:, input_token_len:]
    else:
        # Output contains only new tokens
        generated_ids = output_sequences

    outputs = tokenizer.batch_decode(
        generated_ids.cpu(), skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    return attention_output.detach(), outputs

def get_model(model_path = "llava-v1.5-7b"):
    model_path = f"liuhaotian/{model_path}"
    model_path = model_path
    model_base = None
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
    )
    return tokenizer, model, image_processor, context_len, model_name

if __name__ == "__main__":
    prompt = "What are the things I should be cautious about when I visit here?"
    image_file = "https://llava-vl.github.io/static/images/view.jpg"

    tokenizer, model, image_processor, context_len = get_model()

    args = type('Args', (), {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "context_len": context_len,
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
    })()

    mask = getmask(args)