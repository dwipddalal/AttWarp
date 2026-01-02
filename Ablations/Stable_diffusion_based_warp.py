# warp_attention.py

import sys
import numpy as np
import cv2
from PIL import Image

# --- Transform definitions & registry ---

def identity_transform(x):
    # print(f"[warp] identity_transform input shape: {x.shape}")
    return x

def identity_inverse(x):
    return x

def square_transform(x):
    # print(f"[warp] square_transform input shape: {x.shape}")
    return x**2

def square_inverse(x):
    return np.sqrt(np.maximum(x, 0))

def sqrt_transform(x):
    # print(f"[warp] sqrt_transform input shape: {x.shape}")
    return np.sqrt(np.maximum(x, 0))

def sqrt_inverse(x):
    return x**2

def exp_transform(x):
    # print(f"[warp] exp_transform input shape: {x.shape}")
    return np.exp(EXP_SCALE * x) / EXP_DIVISOR

def exp_inverse(x):
    return np.log(np.maximum(x * EXP_DIVISOR, 1e-9)) / EXP_SCALE

def log_transform(x):
    # print(f"[warp] log_transform input shape: {x.shape}")
    return np.log(x + 1e-5)

def log_inverse(x):
    return np.exp(x) - 1e-5

INVERSE_TRANSFORMS = {
    identity_transform: identity_inverse,
    square_transform:   square_inverse,
    sqrt_transform:     sqrt_inverse,
    exp_transform:      exp_inverse,
    log_transform:      log_inverse,
}

# Global parameters (tunable)
EPSILON = 1e-9
BASE_ATTENTION = 1e-9
EXP_SCALE = 1.0
EXP_DIVISOR = 1.0
APPLY_INVERSE_TO_MARGINALS = False
ATTENTION_TRANSFORM = identity_transform

def set_transform_function(transform_name, exp_scale=1.0, exp_divisor=1.0, apply_inverse=False):
    global ATTENTION_TRANSFORM, EXP_SCALE, EXP_DIVISOR, APPLY_INVERSE_TO_MARGINALS
    EXP_SCALE = exp_scale
    EXP_DIVISOR = exp_divisor
    APPLY_INVERSE_TO_MARGINALS = apply_inverse
    # print(f"[warp] set_transform_function → name={transform_name}, exp_scale={exp_scale}, exp_divisor={exp_divisor}, apply_inverse={apply_inverse}")
    if transform_name == "identity":
        ATTENTION_TRANSFORM = identity_transform
    elif transform_name == "square":
        ATTENTION_TRANSFORM = square_transform
    elif transform_name == "sqrt":
        ATTENTION_TRANSFORM = sqrt_transform
    elif transform_name == "exp":
        ATTENTION_TRANSFORM = exp_transform
    elif transform_name == "log":
        ATTENTION_TRANSFORM = log_transform
    else:
        # print(f"[warp] Unknown transform `{transform_name}`, falling back to identity")
        ATTENTION_TRANSFORM = identity_transform
    return transform_name

def resize_image_to_match_attmap(image, att_map):
    # print(f"[warp] resize_image_to_match_attmap: image.shape={image.shape}, att_map.shape={att_map.shape}")
    ih, iw = image.shape[:2]
    ah, aw = att_map.shape[:2]
    if (ih, iw) == (ah, aw):
        # print("[warp] shapes already match, returning copy")
        return image.copy()
    # print(f"[warp] resizing image from {(ih,iw)} to {(ah,aw)}")
    resized = cv2.resize(image, (aw, ah), interpolation=cv2.INTER_LINEAR)
    return resized

def warp_image_by_attention(image, att_map, new_width, new_height):
    # print(f"[warp] warp_image_by_attention called with image.shape={image.shape}, att_map.shape={att_map.shape}, target=({new_width},{new_height})")
    if image is None or att_map is None:
        # print("[warp] ERROR: got None")
        return None

    h, w = image.shape[:2]
    att = att_map.astype(np.float64)
    att = np.maximum(att, 0)

    # apply transform
    att_t = ATTENTION_TRANSFORM(att)
    att_biased = att_t + BASE_ATTENTION

    # marginal profiles
    px = np.sum(att_biased, axis=0)  # (w,)
    py = np.sum(att_biased, axis=1)  # (h,)

    # optional inverse
    if APPLY_INVERSE_TO_MARGINALS and ATTENTION_TRANSFORM in INVERSE_TRANSFORMS:
        inv = INVERSE_TRANSFORMS[ATTENTION_TRANSFORM]
        px = inv(px - BASE_ATTENTION * h) + BASE_ATTENTION * h
        py = inv(py - BASE_ATTENTION * w) + BASE_ATTENTION * w

    sx = np.sum(px)
    sy = np.sum(py)
    if sx < EPSILON or sy < EPSILON:
        # print("[warp] warning: near-zero total attention, using uniform profiles")
        px = np.ones_like(px)
        py = np.ones_like(py)
        sx, sy = px.sum(), py.sum()

    # cumulative → maps
    cx = np.concatenate(([0], np.cumsum(px) / sx)) * new_width
    ox = np.concatenate(([0], np.arange(w)+1))
    cy = np.concatenate(([0], np.cumsum(py) / sy)) * new_height
    oy = np.concatenate(([0], np.arange(h)+1))
    cx[-1], cy[-1] = new_width, new_height

    tx = np.interp(np.arange(new_width), cx, ox).astype(np.float32)
    ty = np.interp(np.arange(new_height), cy, oy).astype(np.float32)
    mx, my = np.meshgrid(tx, ty)

    warped = cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if warped.shape[:2] != (new_height, new_width):
        warped = cv2.resize(warped, (new_width,new_height), interpolation=cv2.INTER_LINEAR)
    # print(f"[warp] warped result shape={warped.shape}")
    return warped

def save_warped_image(image_path, att_map, output_path,
                      width=500, height=500,
                      transform="identity", exp_scale=1.0,
                      exp_divisor=1.0, apply_inverse=False, skip_map_resize=False):
    
    # ─── load input ────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    # ─── upsample the 24×24 map to the image’s H×W ───────────────────
    H, W = img.shape[:2]
    # Option A: use your existing helper (resizes + Gaussian‐blurs + normalizes)
    # att_map_full = upsample_attention_map(att_map, (H, W))
    #
    # Option B: if you want just a plain cubic resize (no blur/normalize),
    # uncomment the two lines below instead of Option A:
    att_map_full = cv2.resize(att_map, (W, H), interpolation=cv2.INTER_CUBIC)
    att_map_full -= att_map_full.min(); att_map_full /= (att_map_full.max() + 1e-9)
    
    # ─── set the transform & warp ──────────────────────────────────
    set_transform_function(transform, exp_scale, exp_divisor, apply_inverse)
    warped = warp_image_by_attention(img, att_map_full, width, height)

    # ─── save ──────────────────────────────────────────────────────
    if warped is not None:
        cv2.imwrite(output_path, warped)
        return True
    else:
        return False



def save_warped_image_original(image_path, att_map, output_path,
                      width=500, height=500,
                      transform="identity", exp_scale=1.0,
                      exp_divisor=1.0, apply_inverse=False, skip_map_resize=False):
    
    # print(f"[warp] save_warped_image: loading `{image_path}`")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    # print(f"[warp] original image shape={img.shape}")
    
    # resize to attention map if needed
    img_resized = resize_image_to_match_attmap(img, att_map)
    
    # set parameters
    set_transform_function(transform, exp_scale, exp_divisor, apply_inverse)

    # perform warp
    warped = warp_image_by_attention(img_resized, att_map, width, height)
    if warped is not None:
        cv2.imwrite(output_path, warped)
        # print(f"[warp] saved warped image → {output_path}")
        return True
    else:
        # print("[warp] warp failed")
        return False


# sd_warp.py

import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from diffusers import DiffusionPipeline, DDIMScheduler
from attention_map_diffusers import init_pipeline, attn_maps
import cv2

# -----------------------------
# 1. Pipeline & scheduler load
# -----------------------------
def load_pipeline(model_id="stabilityai/stable-diffusion-2-1", device="cuda"):
    print(f"[main] Loading DDIMScheduler from `{model_id}`")
    sched = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    print(f"[main] Loading DiffusionPipeline from `{model_id}`")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        scheduler=sched,
        torch_dtype=torch.float16
    ).to(device)
    print("[main] Patching pipeline for attention hooks")
    pipe = init_pipeline(pipe)
    return pipe

# ---------------------------------
# 2. Image → latent encoding (z0)
# ---------------------------------
def encode_image(pipe, image_path):
    # print(f"[main] Open & preprocess `{image_path}`")
    img = Image.open(image_path).convert("RGB")
    tr = Compose([ Resize((768,768)), ToTensor(), Normalize([0.5]*3, [0.5]*3) ])
    x = tr(img).unsqueeze(0).to(pipe.device).to(pipe.vae.dtype)
    dist = pipe.vae.encode(x).latent_dist
    z0 = dist.sample() * pipe.vae.config.scaling_factor
    # print(f"[main] Encoded latent z0 shape={z0.shape}")
    return z0

# ---------------------------------
# 3. Prompt → text embeddings + tokens
# ---------------------------------
def get_text_embeddings(pipe, prompt):
    # print(f"[main] Tokenizing prompt: '{prompt}'")
    toks = pipe.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    )
    input_ids = toks.input_ids.to(pipe.device)
    mask      = toks.attention_mask.to(pipe.device)
    emb       = pipe.text_encoder(input_ids, attention_mask=mask).last_hidden_state
    # print(f"[main] Text embeddings shape={emb.shape}")
    return emb, input_ids

# ----------------------------------------------------
# 4. Truncated DDIM inversion to collect attn_maps
# ----------------------------------------------------
def truncated_ddim_inversion(pipe, latents, text_embeds, timesteps):
    # print(f"[main] Starting truncated DDIM inversion w/ timesteps {timesteps}")
    sched = pipe.scheduler
    max_t = sched.config.num_train_timesteps - 1
    ts = [min(max(int(t),0), max_t) for t in timesteps]
    # print(f"[main] Clamped timesteps → {ts}")
    inv = {}
    z = latents
    for prev_t, t in zip(ts[:-1], ts[1:]):
        # print(f"[main] Inversion step: {prev_t} → {t}, z shape={z.shape}")
        with torch.no_grad():
            out = pipe.unet(
                z,
                torch.tensor([prev_t], device=pipe.device),
                encoder_hidden_states=text_embeds
            ).sample
        a_prev = sched.alphas_cumprod[prev_t]
        a      = sched.alphas_cumprod[t]
        x0     = (z - torch.sqrt(1 - a_prev)*out) / torch.sqrt(a_prev)
        z      = torch.sqrt(a)*x0 + torch.sqrt(1 - a)*out
        inv[t] = z
    # print(f"[main] Completed inversion; snapshots: {list(inv.keys())}")
    return inv

# ----------------------------------------------------
# 5. Forward denoise passes to refill attn_maps
# ----------------------------------------------------
def forward_and_capture(pipe, inv_latents, prompt):
    # print(f"[main] Clearing previous attn_maps and running forward passes")
    attn_maps.clear()
    with torch.no_grad():
        for t, z in inv_latents.items():
            # print(f"[main] Forward denoising at t={t}, latent shape={z.shape}")
            _ = pipe(
            prompt=[prompt],
            negative_prompt=[""],
            num_inference_steps=10,
            latents=z,
            output_type="latent"
            )
            # print(f"[main] Captured maps for timesteps: {list(attn_maps.keys())}")

# ----------------------------------------------------
# 6a. Compute per-token importance across all maps
# ----------------------------------------------------
def compute_token_importance(
    captured_maps,
    inference_step_selection=None,
    num_layers=None,
    last_timestep=False
):
    """
    Compute per-token importance scores from captured attention maps.
    
    Args:
      captured_maps: dict[timestep -> dict[layer_key -> Tensor[B,heads,H,W,tokens]]].
      inference_step_selection: optional list of timesteps to include. None = all.
      num_layers: optional int, number of last layers per timestep to include. 
                  None = all layers.
      last_timestep: if True, only use the single final timestep and its final layer.
    
    Returns:
      1D numpy array of length = number of tokens, the (mean) importance scores.
    """
    import numpy as np

    # Special case: just the very last timestep & layer
    if last_timestep:
        t = max(captured_maps.keys())
        layer_dict = captured_maps[t]
        k = max(layer_dict.keys())
        attn = layer_dict[k]  # shape [B,heads,H,W,tokens]
        return attn.sum(dim=(0,1,2,3)).cpu().numpy()

    # Otherwise build a list of per-layer per-timestep score vectors
    timesteps = (
        [t for t in captured_maps if t in inference_step_selection]
        if inference_step_selection is not None
        else list(captured_maps.keys())
    )
    token_scores = []

    for t in timesteps:
        layer_dict = captured_maps[t]
        keys = sorted(layer_dict.keys())
        if num_layers is not None:
            keys = keys[-num_layers:]
        for k in keys:
            attn = layer_dict[k]     # [B, heads, H, W, tokens]
            scores = attn.sum(dim=(0,1,2,3)).cpu().numpy()
            token_scores.append(scores)

    if not token_scores:
        raise ValueError("No attention maps to compute token importance")

    return np.mean(token_scores, axis=0)

# ----------------------------------------------------
# 6b. Aggregate only top-k token channels into a 2D map
# ----------------------------------------------------
def aggregate_attention_map_for_tokens(captured_maps, token_indices, inference_step_selection=None,num_last_layers=1):
    maps_2d = []
    # Get all layer keys sorted to access last 5 maps
    all_steps = sorted(list(captured_maps.keys()))
    
    # Select steps based on inference_step_selection
    if inference_step_selection is not None:
        steps_to_use = [step for step in all_steps if step in inference_step_selection]
    else:
        # If no selection provided, use all steps
        steps_to_use = all_steps
    # Take only last num_last_layers attention maps for each selected step
    for step in steps_to_use:
        layer_dict = captured_maps[step] 
        layer_keys = sorted(list(layer_dict.keys()))[-num_last_layers:]  # Get last num_last_layers
        for key in layer_keys:
            attn = layer_dict[key]
            # select just our specified token channels
            attn_sel = attn[..., token_indices]  # [B, heads, H, W, k]
            m = attn_sel.sum(dim=0).sum(dim=0).sum(dim=-1)  # → [H, W]
            maps_2d.append(m.detach().cpu().numpy().astype(np.float32))
            
    if not maps_2d:
        raise ValueError("No attention maps to aggregate")
        
    heights, widths = zip(*(m.shape for m in maps_2d))
    max_h, max_w = max(heights), max(widths)
    resized = []
    
    for m in maps_2d:
        if m.shape != (max_h, max_w):
            m = cv2.resize(m, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        resized.append(m)
        
    att_map_lr = np.stack(resized, axis=0).mean(axis=0)
    att_map_lr -= att_map_lr.min()
    att_map_lr /= (att_map_lr.max() + 1e-9)
    return att_map_lr

# ----------------------------------------------------
# 7. Upsample to full-res
# ----------------------------------------------------
def upsample_attention_map(att_map_lr, target_size):
    H, W = target_size
    att_map_full = cv2.resize(att_map_lr, (W, H), interpolation=cv2.INTER_CUBIC)
    att_map_full = cv2.GaussianBlur(att_map_full, ksize=(5,5), sigmaX=1.0)
    att_map_full -= att_map_full.min()
    att_map_full /= (att_map_full.max() + 1e-9)
    return att_map_full

# ----------------------------------------------------
# 8. Main run loop
# ----------------------------------------------------
def run(
    image_path, prompt, output_warp,
    width, height, transform,
    exp_scale, exp_divisor, apply_inverse,
    iterations, top_k
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(device=device)

    # encode & invert
    z0 = encode_image(pipe, image_path)
    emb, input_ids = get_text_embeddings(pipe, prompt)
    inv = truncated_ddim_inversion(pipe, z0, emb, [1000,800,600,400,200])
    forward_and_capture(pipe, inv, prompt)

    # token importance → top-k
    mean_scores = compute_token_importance(attn_maps,inference_step_selection=None)
    top_indices = list(np.argsort(mean_scores)[-top_k:][::-1])
    tok_strs = pipe.tokenizer.convert_ids_to_tokens(input_ids[0])
    # print(f"[main] Top {top_k} tokens by attention: {[tok_strs[i] for i in top_indices]}")

    # build attention map from those tokens only
    att_map_lr = aggregate_attention_map_for_tokens(attn_maps, top_indices,inference_step_selection=None,num_last_layers=1)
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    att_map = upsample_attention_map(att_map_lr, (H, W))

    # iterative warping
    step_scale = exp_scale / max(1, iterations)
    cur_input = image_path
    for i in range(iterations):
        out_path = output_warp if i == iterations-1 else f"{os.path.splitext(output_warp)[0]}_it{i+1}.png"
        save_warped_image(
            cur_input, att_map, out_path,
            width, height,
            transform, step_scale, exp_divisor, apply_inverse,
            skip_map_resize=True
        )
        cur_input = out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",       default="/home/skan/test_images/china.png")
    p.add_argument("--prompt",      default="What country is one book about? Answer the question using a single word or phrase.")
    p.add_argument("--output_warp", default="/home/skan/new_code_neurips/experimentTokens/results/sd_warped_china_AIS_TS_AIS_AT2.png")
    p.add_argument("--width",       type=int, default=500)
    p.add_argument("--height",      type=int, default=500)
    p.add_argument("--transform",   choices=["identity","square","sqrt","exp","log"], default="identity")
    p.add_argument("--exp_scale",   type=float, default=1.0)
    p.add_argument("--exp_divisor", type=float, default=1.0)
    p.add_argument("--apply_inverse", action="store_true")
    p.add_argument("--iterations",  type=int, default=1,
                   help="How many times to apply the warp iteratively")
    p.add_argument("--top_k",       type=int, default=20,
                   help="Number of top tokens to use for attention aggregation")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_warp) or ".", exist_ok=True)
    run(
        args.image, args.prompt, args.output_warp,
        args.width, args.height, args.transform,
        args.exp_scale, args.exp_divisor, args.apply_inverse,
        args.iterations, args.top_k
    )
