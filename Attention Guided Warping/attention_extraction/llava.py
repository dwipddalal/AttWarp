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
import torchvision.transforms as T

# Import constants from llava for image token handling
try:
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    IMAGE_TOKEN_INDEX = -200  # Default fallback

# Import getmask and get_model from the local functions module
# Use relative import when used as a package, absolute when run standalone
try:
    from .functions import getmask, get_model
except ImportError:
    # Fallback for when this file is run directly or imported from parent
    import os as _os
    import sys as _sys
    _this_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _this_dir not in _sys.path:
        _sys.path.insert(0, _this_dir)
    from functions import getmask, get_model


class MaskHookLogger(object):
    """
    Hook logger that captures attention weights from a specific layer during generation.
    Uses PyTorch's native register_forward_hook mechanism.
    """
    def __init__(self, model, device, layer_index=20):
        self.device = device
        self.attns = []
        self.model = model
        self.layer_index = layer_index
        self.hook_handle = None
        self.image_token_start = None
        self.image_token_end = None
        self.num_image_tokens = 576  # 24x24 patches for LLaVA-1.5

    def _find_image_token_range(self, input_ids):
        """Find the range of image tokens in the input sequence."""
        # In LLaVA, image tokens are inserted at the position of IMAGE_TOKEN_INDEX
        # After processing, they become 576 tokens (24x24 patches)
        input_ids_list = input_ids[0].tolist()

        # Find where image tokens start - they replace the IMAGE_TOKEN placeholder
        # In the actual input_ids during generation, image tokens are already expanded
        # We need to find them based on the pattern

        # For LLaVA-1.5, image tokens are typically at the beginning after the system prompt
        # A simpler heuristic: image tokens are the first 576 tokens after position 0
        # But more accurately, we should track where they were inserted

        # Default: assume image tokens start after the first few tokens (typically <s> and possibly <image>)
        # and span 576 tokens
        start = 1  # Skip BOS token
        end = start + self.num_image_tokens

        return start, end

    @torch.no_grad()
    def _attention_hook(self, module, input, output):
        """
        Forward hook to capture attention weights.
        For LlamaAttention, the output is (hidden_states, attention_weights, past_key_value)
        when output_attentions=True, or just (hidden_states, past_key_value) otherwise.

        We need to capture the attention weights directly from the computation.
        """
        # The hook receives the output of the attention layer
        # For LlamaAttention with output_attentions=True, output is a tuple
        # (attn_output, attn_weights, past_key_value)

        if isinstance(output, tuple) and len(output) >= 2:
            # Check if second element looks like attention weights
            attn_weights = output[1]
            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                if len(attn_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    self._process_attention(attn_weights)

    @torch.no_grad()
    def _process_attention(self, attn_weights):
        """Process attention weights to extract image attention."""
        # attn_weights shape: [batch, num_heads, seq_len, seq_len]
        # We want attention from the last token to image tokens

        if self.image_token_start is None or self.image_token_end is None:
            # Use default range
            st = 1
            ed = min(1 + self.num_image_tokens, attn_weights.shape[-1])
        else:
            st = self.image_token_start
            ed = min(self.image_token_end, attn_weights.shape[-1])

        # Get attention from last generated token to image tokens
        # Shape: [batch, num_heads, num_image_tokens]
        image_attention = attn_weights[:, :, -1, st:ed].detach()

        # Normalize the sliced attention to sum to 1 per head (re-normalize after slicing)
        # Note: attn_weights are already post-softmax, so do NOT apply softmax again
        image_attention = image_attention / (image_attention.sum(dim=-1, keepdim=True) + 1e-12)
        image_attention = image_attention.mean(dim=1)  # Average over heads

        self.attns.append(image_attention)  # [batch, num_image_tokens]

    def set_image_token_range(self, start, end):
        """Set the range of image tokens in the input sequence."""
        self.image_token_start = start
        self.image_token_end = end

    @torch.no_grad()
    def finalize(self):
        """Finalize and return the aggregated attention map."""
        if len(self.attns) == 0:
            # Return uniform attention if no attention was captured
            return torch.ones(self.num_image_tokens, device=self.device) / self.num_image_tokens

        # Stack all attention maps and average
        attns = torch.cat(self.attns, dim=0).to(self.device)
        return attns.mean(dim=0)

    def reinit(self):
        """Reinitialize the logger for a new forward pass."""
        self.attns = []
        self.image_token_start = None
        self.image_token_end = None
        torch.cuda.empty_cache()

    def register_hook(self):
        """Register the forward hook on the attention layer."""
        if self.hook_handle is not None:
            self.hook_handle.remove()

        attn_layer = self.model.model.layers[self.layer_index].self_attn
        self.hook_handle = attn_layer.register_forward_hook(self._attention_hook)

    def remove_hook(self):
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def hook_logger(model, device, layer_index=20):
    """
    Create and register a hook logger for attention extraction.

    This function patches the model to output attention weights during generation
    and registers a forward hook to capture them.

    Args:
        model: The LLaVA model
        device: The device to use
        layer_index: Which transformer layer to hook (default: 20)

    Returns:
        MaskHookLogger instance
    """
    # Create the hook logger
    prs = MaskHookLogger(model, device, layer_index)

    # Store original config value
    original_output_attentions = getattr(model.config, 'output_attentions', False)

    # Enable attention output in the model config
    model.config.output_attentions = True

    # Register the forward hook
    prs.register_hook()

    # Store reference in model for easy access
    model.hooklogger = prs
    model._original_output_attentions = original_output_attentions

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

def llava_api(images, queries, model_name, batch_size = 1, layer_index = 20, enhance_coe = 10, kernel_size = 3, interpolate_method_name = "LANCZOS", grayscale = 0):

    """
    Generates image masks and blends them using the specified model and parameters.

    Parameters:
    images (list): list of images. Each item can be a path to image (str) or a PIL.Image. 
    queries (list): list of queries. Each item is a str. 
    batch_size (int): Batch size for processing images. Only support 1.
    model_name (str): Name of the model to load the pretrained model. One of "llava-v1.5-7b" and "llava-v1.5-13b".
    layer_index (int): Index of the layer in the model to hook. Default is 20 (mid-layer).
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
            # Reset hook logger for each new image/question
            hl.reinit()

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


# ────────────────────────────────────────────────────────────────────────────
# Batched attention extraction
# ────────────────────────────────────────────────────────────────────────────

class BatchMaskHookLogger(object):
    """
    Hook logger for batched attention extraction from a specific layer.
    Handles per-sample image token ranges (different due to padding offsets).
    Monkey-patches only the target layer to force output_attentions=True,
    so all other layers keep using fast SDPA / default attention.
    """

    def __init__(self, model, device, layer_index=20):
        self.device = device
        self.model = model
        self.layer_index = layer_index
        self.hook_handle = None
        self.num_image_tokens = 576  # 24x24 patches for LLaVA-1.5

        # Per-sample state
        self.image_token_starts = None  # List[int]
        self.image_token_ends = None    # List[int]
        self.batch_size = 0

        # Accumulated attention per generation step: list of [batch, 576] tensors
        self.step_attentions = []

        # Monkey-patch bookkeeping
        self._original_forward = None

    def set_batch_image_token_ranges(self, starts, ends):
        """Set per-sample image token start/end positions."""
        assert len(starts) == len(ends)
        self.image_token_starts = starts
        self.image_token_ends = ends
        self.batch_size = len(starts)

    # ── hook callback ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _attention_hook(self, module, input, output):
        if not isinstance(output, tuple) or len(output) < 2:
            return
        attn_weights = output[1]
        if attn_weights is None or not isinstance(attn_weights, torch.Tensor):
            return
        if len(attn_weights.shape) != 4:  # [batch, heads, q_len, kv_len]
            return
        self._process_attention(attn_weights)

    @torch.no_grad()
    def _process_attention(self, attn_weights):
        bsz = attn_weights.shape[0]
        per_sample = []
        for b in range(bsz):
            st = self.image_token_starts[b]
            ed = min(self.image_token_ends[b], attn_weights.shape[-1])
            # Last query token's attention to image tokens
            img_attn = attn_weights[b, :, -1, st:ed].detach()  # [heads, img_tokens]
            # Re-normalize sliced attention (already post-softmax, do NOT apply softmax again)
            img_attn = (img_attn / (img_attn.sum(dim=-1, keepdim=True) + 1e-12)).mean(dim=0)  # [img_tokens]
            per_sample.append(img_attn)
        self.step_attentions.append(torch.stack(per_sample, dim=0))  # [bsz, 576]

    # ── finalize ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def finalize_batch(self):
        """Return list of [24, 24] attention maps, one per sample."""
        if len(self.step_attentions) == 0:
            return [
                torch.ones(self.num_image_tokens, device=self.device)
                / self.num_image_tokens
                for _ in range(self.batch_size)
            ]
        all_steps = torch.stack(self.step_attentions, dim=0)  # [steps, bsz, 576]
        avg = all_steps.mean(dim=0)  # [bsz, 576]
        return [avg[i].view(24, 24) for i in range(self.batch_size)]

    # ── lifecycle ──────────────────────────────────────────────────────────

    def reinit(self):
        self.step_attentions = []
        self.image_token_starts = None
        self.image_token_ends = None
        self.batch_size = 0
        torch.cuda.empty_cache()

    def register_hook_and_patch(self):
        """Register hook AND monkey-patch target layer to force output_attentions."""
        if self.hook_handle is not None:
            self.hook_handle.remove()

        attn_layer = self.model.model.layers[self.layer_index].self_attn
        self.hook_handle = attn_layer.register_forward_hook(self._attention_hook)

        # Monkey-patch: force output_attentions=True for this layer only
        self._original_forward = attn_layer.forward
        _orig = self._original_forward

        def _patched_forward(*args, **kwargs):
            kwargs["output_attentions"] = True
            return _orig(*args, **kwargs)

        attn_layer.forward = _patched_forward

    def remove_hook_and_unpatch(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        if self._original_forward is not None:
            self.model.model.layers[self.layer_index].self_attn.forward = (
                self._original_forward
            )
            self._original_forward = None


def batch_hook_logger(model, device, layer_index=20):
    """
    Create and register a BatchMaskHookLogger.

    Unlike hook_logger(), this does NOT set model.config.output_attentions = True.
    Only the target layer is patched to produce attention weights.
    """
    prs = BatchMaskHookLogger(model, device, layer_index)
    model.config.output_attentions = False
    prs.register_hook_and_patch()
    model.batch_hooklogger = prs
    return prs