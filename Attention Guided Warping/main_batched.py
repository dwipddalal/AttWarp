"""
Batched TextVQA processing with LLaVA attention extraction.
Uses batched model.generate() for better GPU utilization.
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
import json
from tqdm import tqdm
import signal
import pickle
import atexit
import torchvision.transforms as T

from new_method import save_warped_image

# Ensure official LLaVA package is importable BEFORE adding attention_extraction
_CLIP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAVA_DIR = os.path.join(_CLIP_ROOT, "LLaVA")
if os.path.isdir(LLAVA_DIR) and LLAVA_DIR not in sys.path:
    sys.path.insert(0, LLAVA_DIR)

ATTN_EXTRACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention_extraction")

from attention_extraction.functions import getmask_batch, get_model

# Import batch_hook_logger and blend_mask from the local llava.py without shadowing 'llava' package
import importlib.util as _importlib_util
_LLAVA_UTIL_PATH = os.path.join(ATTN_EXTRACT_DIR, "llava.py")
_spec = _importlib_util.spec_from_file_location("ae_llava_util", _LLAVA_UTIL_PATH)
ae_llava_util = _importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ae_llava_util)
batch_hook_logger = ae_llava_util.batch_hook_logger
blend_mask = ae_llava_util.blend_mask

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# ── Configuration ──────────────────────────────────────────────────────────
BATCH_SIZE = 32
MODEL_NAME = "llava-v1.5-7b"
TEXTVQA_JSON_PATH = '/shared/nas2/dwip2/data/dataloader/textvqa/TextVQA_0.5.1_val.json'
TEXTVQA_IMAGE_DIR = '/shared/nas2/dwip2/data/dataloader/textvqa/train_images'

OUTPUT_BASE_DIR = "/shared/nas2/dwip2/CLIP/results2/textvqa_processed_batched"
ATTENTION_MAPS_DIR     = os.path.join(OUTPUT_BASE_DIR, "attention_maps")
WARPED_IMAGES_DIR      = os.path.join(OUTPUT_BASE_DIR, "warped_images")
ORIGINAL_IMAGES_DIR    = os.path.join(OUTPUT_BASE_DIR, "original_images")
METADATA_DIR           = os.path.join(OUTPUT_BASE_DIR, "metadata")
MASKED_IMAGES_DIR      = os.path.join(OUTPUT_BASE_DIR, "masked_images")
ATTENTION_MAPS_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "attention_maps_images")
RAW_ATTENTION_MAPS_DIR = os.path.join(OUTPUT_BASE_DIR, "raw_attention_maps")
CHECKPOINT_DIR         = os.path.join(OUTPUT_BASE_DIR, "checkpoints")

for d in [OUTPUT_BASE_DIR, ATTENTION_MAPS_DIR, WARPED_IMAGES_DIR,
          ORIGINAL_IMAGES_DIR, METADATA_DIR, MASKED_IMAGES_DIR,
          ATTENTION_MAPS_IMAGES_DIR, RAW_ATTENTION_MAPS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500

# ── Dataset (reused from main.py) ─────────────────────────────────────────
from torch.utils.data import Dataset

class TextVQADataset(Dataset):
    def __init__(self, json_path, image_dir=None):
        self.image_dir = image_dir
        print(f"Loading TextVQA data from {json_path}...")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: {e}")
            self.samples = []
            return
        self.samples = data.get('data', [])
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        image = self._get_image(sample)
        sample['loaded_image'] = image
        return sample

    def _get_image(self, sample):
        image_id = sample.get('image_id')
        if not image_id or not self.image_dir:
            return None
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if os.path.exists(img_path):
            try:
                return Image.open(img_path).convert('RGB')
            except Exception:
                return None
        return None


# ── Checkpointing ─────────────────────────────────────────────────────────
def save_checkpoint(data, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Checkpoint save error: {e}")

def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    dataset = TextVQADataset(json_path=TEXTVQA_JSON_PATH, image_dir=TEXTVQA_IMAGE_DIR)

    # Prepare all valid samples
    all_images, all_questions, all_meta = [], [], []
    print("Preparing samples...")
    for idx in tqdm(range(len(dataset)), desc="Loading"):
        entry = dataset[idx]
        image = entry.get('loaded_image')
        question = entry.get('question')
        if image is None or not question:
            continue
        all_images.append(image)
        all_questions.append(question)
        meta = {k: v for k, v in entry.items() if k != 'loaded_image'}
        meta['original_index'] = idx
        all_meta.append(meta)

    num_total = len(all_images)
    print(f"Prepared {num_total} valid samples")
    if num_total == 0:
        return 0

    # Checkpoint
    ckpt_file = os.path.join(CHECKPOINT_DIR, "batched_checkpoint.pkl")
    ckpt = load_checkpoint(ckpt_file)
    processed_set = ckpt.get('processed', set()) if ckpt else set()
    processed_count = ckpt.get('processed_count', 0) if ckpt else 0
    failed_count = ckpt.get('failed_count', 0) if ckpt else 0

    remaining = [(i, all_images[i], all_questions[i], all_meta[i])
                 for i in range(num_total) if i not in processed_set]

    if not remaining:
        print("All items already processed.")
        return 0

    print(f"Processing {len(remaining)} remaining items (batch_size={BATCH_SIZE})")

    # Load model
    tokenizer, model, image_processor, context_len, inner_model_name = get_model(MODEL_NAME)
    batch_hl = batch_hook_logger(model, model.device, layer_index=20)

    # Checkpoint auto-save
    ckpt_data = {'processed': processed_set, 'processed_count': processed_count,
                 'failed_count': failed_count}

    def save_current():
        save_checkpoint(ckpt_data, ckpt_file)

    atexit.register(save_current)
    def sig_handler(sig, frame):
        print("\nInterrupted. Saving checkpoint...")
        save_current()
        sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Visualisation params
    enhance_coe = 10
    kernel_size = 3
    interpolate_method = Image.LANCZOS
    grayscale = 0

    # Process in batches
    num_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(0, len(remaining), BATCH_SIZE),
                desc="Batched inference", total=num_batches)

    for batch_start in pbar:
        batch_items = remaining[batch_start: batch_start + BATCH_SIZE]
        b_indices  = [item[0] for item in batch_items]
        b_images   = [item[1] for item in batch_items]
        b_questions = [item[2] for item in batch_items]
        b_meta     = [item[3] for item in batch_items]

        current_bs = len(b_images)

        try:
            attn_maps, output_texts = getmask_batch(
                images=b_images,
                questions=b_questions,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                context_len=context_len,
                model_name=MODEL_NAME,
                batch_hl=batch_hl,
                max_new_tokens=20,
                temperature=0,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\nOOM with batch_size={current_bs}. Processing this batch one-by-one.")
            # Fallback: process each sample individually
            attn_maps, output_texts = [], []
            for img, q in zip(b_images, b_questions):
                try:
                    maps, texts = getmask_batch(
                        images=[img], questions=[q],
                        model=model, tokenizer=tokenizer,
                        image_processor=image_processor,
                        context_len=context_len, model_name=MODEL_NAME,
                        batch_hl=batch_hl, max_new_tokens=20, temperature=0,
                    )
                    attn_maps.append(maps[0])
                    output_texts.append(texts[0])
                except Exception as e:
                    print(f"  Single-sample fallback failed: {e}")
                    attn_maps.append(torch.ones(24, 24, device=model.device) / 576)
                    output_texts.append("")
        except Exception as e:
            print(f"\nBatch error: {e}. Skipping batch.")
            for idx in b_indices:
                ckpt_data['processed'].add(idx)
                ckpt_data['failed_count'] += 1
            save_current()
            continue

        # Save results per sample
        for j in range(current_bs):
            idx = b_indices[j]
            meta = b_meta[j]
            image = b_images[j]
            attn_24 = attn_maps[j]
            out_text = output_texts[j]

            image_id = meta.get('image_id', str(idx))
            sample_id = f"{image_id}_{idx}"
            failed = False

            try:
                # Original image
                image.save(os.path.join(ORIGINAL_IMAGES_DIR, f"{sample_id}_original.png"))

                # Raw attention map
                attn_t = attn_24.unsqueeze(0).unsqueeze(0)  # [1,1,24,24]
                np.save(os.path.join(RAW_ATTENTION_MAPS_DIR, f"{sample_id}_raw_attn.npy"),
                        attn_t.detach().cpu().numpy())

                # Attention map image
                attn_pil = T.ToPILImage()(attn_t.squeeze(0).cpu())
                attn_pil.save(os.path.join(ATTENTION_MAPS_IMAGES_DIR, f"{sample_id}_attn_map_img.png"))

                # Masked / blended image
                merged_image, mota_mask = blend_mask(
                    image, attn_24, enhance_coe, kernel_size, interpolate_method, grayscale)
                merged_image.save(os.path.join(MASKED_IMAGES_DIR, f"{sample_id}_masked.png"))

                # Mota mask visualisation + npy
                if isinstance(mota_mask, Image.Image):
                    mota_gray = mota_mask.convert('L')
                    mota_gray.save(os.path.join(ATTENTION_MAPS_DIR, f"{sample_id}_mota_mask_vis.png"))
                    mota_np = np.array(mota_gray)
                    np.save(os.path.join(ATTENTION_MAPS_DIR, f"{sample_id}_mota_mask.npy"), mota_np)

                    # Warped image
                    save_warped_image(
                        image_path=image, att_map=mota_np,
                        original_image_save_path=None, masked_overlay_save_path=None,
                        output_path=os.path.join(WARPED_IMAGES_DIR, f"{sample_id}_identity.png"),
                        vis_path=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                        transform="identity", exp_scale=1.0, exp_divisor=1.0,
                        apply_inverse=False,
                    )

                # Metadata
                meta["model_output"] = out_text
                meta["sample_id"] = sample_id
                with open(os.path.join(METADATA_DIR, f"{sample_id}_metadata.json"), 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"Error saving {sample_id}: {e}")
                failed = True

            ckpt_data['processed'].add(idx)
            if failed:
                ckpt_data['failed_count'] += 1
            else:
                ckpt_data['processed_count'] += 1

        # Running stats
        pbar.set_postfix({
            "done": ckpt_data['processed_count'],
            "fail": ckpt_data['failed_count'],
        })
        save_current()

    print(f"\nDone. Processed: {ckpt_data['processed_count']}, "
          f"Failed: {ckpt_data['failed_count']}")
    return 0 if ckpt_data['failed_count'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
