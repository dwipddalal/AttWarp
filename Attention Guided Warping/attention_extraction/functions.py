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

def getmask_batch(images, questions, model, tokenizer, image_processor,
                   context_len, model_name, batch_hl,
                   max_new_tokens=20, temperature=0, top_p=None, num_beams=1):
    """
    Batched version of getmask(). Processes multiple image-question pairs
    simultaneously through model.generate().

    Args:
        images: List[PIL.Image]
        questions: List[str]
        model, tokenizer, image_processor, context_len, model_name: model artefacts
        batch_hl: BatchMaskHookLogger instance (already registered on the model)
        max_new_tokens, temperature, top_p, num_beams: generation params

    Returns:
        attention_maps: List[Tensor] each [24,24]
        output_texts: List[str]
    """
    disable_torch_init()
    batch_hl.reinit()
    batch_size = len(images)

    # ── 1. determine conv mode (same for all samples) ──────────────────────
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # ── 2. tokenise each prompt ────────────────────────────────────────────
    all_input_ids = []       # list of 1-D tensors (variable length)
    image_token_positions = []

    for question in questions:
        qs = question
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

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt")          # 1-D
        all_input_ids.append(ids)

        ids_list = ids.tolist()
        try:
            img_pos = ids_list.index(IMAGE_TOKEN_INDEX)
        except ValueError:
            img_pos = 1
        image_token_positions.append(img_pos)

    # ── 3. left-pad to same length ─────────────────────────────────────────
    max_len = max(ids.shape[0] for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, ids in enumerate(all_input_ids):
        seq_len = ids.shape[0]
        padded[i, max_len - seq_len:] = ids
        attn_mask[i, max_len - seq_len:] = 1

    padded = padded.cuda()
    attn_mask = attn_mask.cuda()

    # ── 4. process images ──────────────────────────────────────────────────
    imgs_rgb = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
    images_tensor = process_images(imgs_rgb, image_processor, model.config)
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    # ── 5. pre-compute image token positions after multimodal expansion ────
    # prepare_inputs_labels_for_multimodal replaces 1 IMAGE_TOKEN with 576
    # then re-pads based on model.config.tokenizer_padding_side
    unpadded_lens = [int(attn_mask[i].sum().item()) for i in range(batch_size)]
    expanded_lens = [ul - 1 + 576 for ul in unpadded_lens]
    max_expanded = max(expanded_lens)

    # Ensure left-padding after multimodal expansion (required for generation)
    old_padding_side = getattr(model.config, "tokenizer_padding_side", "right")
    model.config.tokenizer_padding_side = "left"

    img_starts, img_ends = [], []
    for i in range(batch_size):
        pad_offset = max_expanded - expanded_lens[i]
        st = pad_offset + image_token_positions[i]
        img_starts.append(st)
        img_ends.append(st + 576)

    batch_hl.set_batch_image_token_ranges(img_starts, img_ends)

    # ── 6. stopping criteria ───────────────────────────────────────────────
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    # ── 7. batched generate ────────────────────────────────────────────────
    with torch.inference_mode():
        output_ids = model.generate(
            padded,
            images=images_tensor,
            attention_mask=attn_mask,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_attentions=False,       # hook captures what we need
            return_dict_in_generate=False,  # return plain tensor
        )

    # Restore original padding side
    model.config.tokenizer_padding_side = old_padding_side

    # ── 8. extract attention maps ──────────────────────────────────────────
    attention_maps = batch_hl.finalize_batch()

    # ── 9. decode per-sample ───────────────────────────────────────────────
    input_len = padded.shape[1]
    output_texts = []
    for i in range(batch_size):
        gen = output_ids[i, input_len:] if output_ids.shape[1] > input_len else output_ids[i]
        text = tokenizer.decode(gen.cpu(), skip_special_tokens=True).strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()
        output_texts.append(text)

    return attention_maps, output_texts


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