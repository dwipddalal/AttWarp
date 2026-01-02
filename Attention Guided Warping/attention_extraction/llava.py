## Imports
import os, time, argparse, base64, requests, os, json, sys, datetime
from itertools import product
import warnings
warnings.filterwarnings("ignore")

import cv2
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import torchvision.transforms as T

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from attention_extraction.functions import getmask, get_model

from attention_extraction.hook import HookManager

def init_hookmanager(module):
    module.hook_manager = HookManager()

class MaskHookLogger(object):
    def __init__(self, model, device):
        self.current_layer = 0
        self.device = device
        self.attns = []
        self.projected_attns = []
        self.image_embed_range = []
        self.index = []
        self.model = model
        
    @torch.no_grad()
    def compute_attentions(self, ret):
        assert len(self.image_embed_range) > 0
        st, ed = self.image_embed_range[-1]
        image_attention = ret[:,:,-1,st:ed].detach()
        image_attention = image_attention.mean(dim = 1)
        self.attns.append(image_attention) # [b, k]
        return ret
        
    @torch.no_grad()
    def compute_projected_attentions(self, ret):
        assert len(self.image_embed_range) > 0
        st, ed = self.image_embed_range[-1]
        image_attention = ret[:,-1,st:ed].detach() # [b, k, d]
        self.projected_attns.append(image_attention) # [b, k, d]
        return ret
        
    @torch.no_grad()
    def compute_attentions_withsoftmax(self, ret):
        assert len(self.image_embed_range) > 0
        st, ed = self.image_embed_range[-1]
        image_attention = ret[:,:,-1,st:ed].detach()
        image_attention = image_attention.softmax(dim = -1)
        image_attention = image_attention.mean(dim = 1)
        self.attns.append(image_attention) # [b, k]
        return ret
    
    @torch.no_grad()
    def compute_logits_index(self, ret):
        next_token_logits = ret[:, -1, :]
        index = next_token_logits.argmax(dim=-1)
        self.index.append(index.item())
        return ret
    
    @torch.no_grad()
    def finalize(self):
        attns = torch.cat(self.attns, dim = 0).to(self.device) 
        return attns.mean(dim = 0)
    
    @torch.no_grad()
    def finalize_projected_attn(self, norm_weight, proj):
        assert len(self.index) == len(self.projected_attns)
        mask = []
        for i in range(-4,-2):
            index = self.index[i]
            attns = self.projected_attns[i].to(self.device) # 1,k,d
            input_dtype = attns.dtype
            attns_var = attns.to(torch.float32).sum(dim = 1).pow(2).mean(-1, keepdim=True)# 1,d
            attns_var = attns_var.unsqueeze(1)# 1,1,d
            normalized_attns = attns * torch.rsqrt(attns_var + 1e-6) # 1,k,d
            normalized_attns = norm_weight.to(normalized_attns.device) * normalized_attns.to(input_dtype) # 1,k,d
            logits = proj(normalized_attns) 
            max_logits = logits[0,:,index] # k
            mask.append(max_logits)

        mask = torch.stack(mask, dim = 0)

        return mask.mean(dim = 0)
        
    def reinit(self):
        self.attns = []
        self.projected_attns = []
        self.image_embed_range = []
        self.index = []
        torch.cuda.empty_cache()

    def log_image_embeds_range(self, ret):
        self.image_embed_range.append(ret[0][0])
        return ret

def hook_logger(model, device, layer_index = 20):
    """Hooks a projected residual stream logger to the model."""

    init_hookmanager(model.model.layers[layer_index].self_attn)

    prs = MaskHookLogger(model, device)
    model.model.layers[layer_index].self_attn.hook_manager.register('after_attn_mask',
                                prs.compute_attentions_withsoftmax)

    model.hooklogger = prs
    
    return prs

def readImg(p):
    return Image.open(p)

def toImg(t):
    return T.ToPILImage()(t)

def invtrans(mask, image, method = Image.BICUBIC):
    return mask.resize(image.size, method)

def merge(mask, image, grap_scale = 200):
    gray = np.ones((image.size[1], image.size[0], 3))*grap_scale
    image_np = np.array(image).astype(np.float32)[..., :3]
    mask_np = np.array(mask).astype(np.float32)
    mask_np = mask_np / 255.0
    blended_np = image_np * mask_np[:, :, None]  + (1 - mask_np[:, :, None]) * gray
    blended_image = Image.fromarray((blended_np).astype(np.uint8))
    return blended_image

def normalize(mat, method = "max"):
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    else:
        raise NotImplementedError

def enhance(mat, coe=10):
    mat = mat - mat.mean()
    mat = mat / mat.std()
    mat = mat * coe
    mat = torch.sigmoid(mat)
    mat = mat.clamp(0,1)
    return mat

def revise_mask(patch_mask, kernel_size = 3, enhance_coe = 10):

    patch_mask = normalize(patch_mask, "min")
    patch_mask = enhance(patch_mask, coe = enhance_coe)

    assert kernel_size % 2 == 1
    padding_size = int((kernel_size - 1) / 2)
    conv = torch.nn.Conv2d(1,1,kernel_size = kernel_size, padding = padding_size, padding_mode = "replicate", stride = 1, bias = False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size**2
    conv.to(patch_mask.device)

    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = patch_mask

    return mask

def blend_mask(image_path_or_pil_image, mask, enhance_coe, kernel_size, interpolate_method, grayscale):
    mask = revise_mask(mask.float(), kernel_size = kernel_size, enhance_coe = enhance_coe)
    mask = mask.detach().cpu()
    mask = toImg(mask.reshape(1,24,24))

    if isinstance(image_path_or_pil_image, str):
        image = readImg(image_path_or_pil_image)
    elif isinstance(image_path_or_pil_image, Image.Image):
        image = image_path_or_pil_image
    else:
        raise NotImplementedError

    # Resize mask to match image
    mask = invtrans(mask, image, method = interpolate_method)
    # Convert mask to normalized grayscale uint8
    mask_np = np.array(mask.convert("L")).astype(np.float32)
    mask_norm = cv2.normalize(mask_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply Jet colormap to mask
    heatmap_bgr = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
    # Prepare original image as BGR NumPy array
    if isinstance(image_path_or_pil_image, str):
        orig_bgr = cv2.imread(image_path_or_pil_image)
    else:
        orig_np = np.array(image.convert("RGB"))
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    # Determine blending alpha (use grayscale parameter if valid, else default 0.5)
    alpha = grayscale if (isinstance(grayscale, (int, float)) and 0 < grayscale <= 1) else 0.5
    # Overlay the heatmap onto the original image
    overlay_bgr = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    merged_image = Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    return merged_image, mask

def llava_api(images, queries, model_name, batch_size = 1, layer_index = 24, enhance_coe = 10, kernel_size = 3, interpolate_method_name = "LANCZOS", grayscale = 0):

    """
    Generates image masks and blends them using the specified model and parameters.

    Parameters:
    images (list): list of images. Each item can be a path to image (str) or a PIL.Image. 
    queries (list): list of queries. Each item is a str. 
    batch_size (int): Batch size for processing images. Only support 1.
    model_name (str): Name of the model to load the pretrained model. One of "llava-v1.5-7b" and "llava-v1.5-13b".
    layer_index (int): Index of the layer in the model to hook. Default is 20.
    enhance_coe (int): Enhancement coefficient for mask blending. Default is 10.
    kernel_size (int): Kernel size for mask blending. Should be odd numbers. Default is 3.
    interpolate_method_name (str): Name of the interpolation method for image processing. Can be any interpolation method supported by PIL.Image.resize. Default is "LANCZOS".
    grayscale (float): Whether to convert the image to grayscale. Default is 0.

    Returns:
    tuple: A tuple containing two lists:
        - masked_images: A list of the masked images. Each item is a PIL.Image.
        - attention_maps: A list of the attention maps as torch tensors with shape (1, 1, 24, 24).
    """

    tokenizer, model, image_processor, context_len, inner_model_name = get_model(model_name)
    hl = hook_logger(model, model.device, layer_index = layer_index)

    interpolate_method = getattr(Image, interpolate_method_name)
    masked_images = []
    attention_maps = []
    mota_masks = []

    for image_path_or_pil_image, question in tqdm(zip(images, queries), total=len(images)):
        with torch.no_grad():
            mask_args = type('Args', (), {
                    "hl":      hl,
                    "model_name": model_name,
                    "model": model,
                    "tokenizer": tokenizer,
                    "image_processor": image_processor,
                    "context_len": context_len,
                    "query": question,
                    "conv_mode": None,
                    "image_file": image_path_or_pil_image,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 20,
                })()
            mask, output = getmask(mask_args)
            # Convert mask of shape (24, 24) to (1, 1, 24, 24)
            attention_map = mask.clone().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 24, 24)
            attention_maps.append(attention_map)  # Store the reshaped attention map
            merged_image, mota_mask = blend_mask(image_path_or_pil_image, mask, enhance_coe, kernel_size, interpolate_method, grayscale)
            masked_images.append(merged_image)
            mota_masks.append(mota_mask)
            
    return masked_images, attention_maps, mota_masks