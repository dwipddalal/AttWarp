import os
import sys
import numpy as np
from PIL import Image
import torch
import json
from tqdm import tqdm
import pdb
import signal
import pickle
import atexit
import torchvision.transforms as T
from io import BytesIO
import requests
import matplotlib.pyplot as plt # Needed for TextVQADataset visualization method (if used)
from torch.utils.data import Dataset, DataLoader

from new_method import save_warped_image
# Ensure official LLaVA package ('llava') is importable BEFORE adding the
# attention_extraction folder (which contains a local llava.py util) to avoid name clashes.
_CLIP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAVA_DIR = os.path.join(_CLIP_ROOT, "LLaVA")
if os.path.isdir(LLAVA_DIR) and LLAVA_DIR not in sys.path:
    sys.path.insert(0, LLAVA_DIR)

# This script lives in finalizing_the_code/, so attention_extraction is a sibling folder.
ATTN_EXTRACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention_extraction")

# Import as a submodule so that attention_extraction/llava.py never shadows the real `llava` package.
from attention_extraction.functions import getmask, get_model
# Import hook_logger from the local attention_extraction/llava.py without shadowing 'llava' package
import importlib.util as _importlib_util
_LLAVA_UTIL_PATH = os.path.join(ATTN_EXTRACT_DIR, "llava.py")
_spec = _importlib_util.spec_from_file_location("ae_llava_util", _LLAVA_UTIL_PATH)
ae_llava_util = _importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ae_llava_util)
hook_logger = ae_llava_util.hook_logger
blend_mask = ae_llava_util.blend_mask

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

MODEL_NAME = "llava-v1.5-7b" # Or the appropriate model for TextVQA processing
TEXTVQA_JSON_PATH = '/shared/nas2/dwip2/data/dataloader/textvqa/TextVQA_0.5.1_val.json'
TEXTVQA_IMAGE_DIR = '/shared/nas2/dwip2/data/dataloader/textvqa/train_images'
DOWNLOAD_IMAGES = False

# Define base output directory for TextVQA results (match POPE script pattern)
OUTPUT_BASE_DIR = "/shared/nas2/dwip2/CLIP/results/textvqa_processed_my_method"
ATTENTION_MAPS_DIR = os.path.join(OUTPUT_BASE_DIR, "attention_maps")
WARPED_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "warped_images")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, "visualizations") # Keep for potential future use
ORIGINAL_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "original_images")
METADATA_DIR = os.path.join(OUTPUT_BASE_DIR, "metadata")
MASKED_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "masked_images")
ATTENTION_MAPS_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "attention_maps_images")
# New directory for raw attention maps (e.g., 24x24 tensors)
RAW_ATTENTION_MAPS_DIR = os.path.join(OUTPUT_BASE_DIR, "raw_attention_maps")
CHECKPOINT_DIR = os.path.join(OUTPUT_BASE_DIR, "checkpoints")

# Create directories (without category subfolders)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TYPE_DIRECTORIES = [OUTPUT_BASE_DIR, ATTENTION_MAPS_DIR, WARPED_IMAGES_DIR, VISUALIZATIONS_DIR,
                    ORIGINAL_IMAGES_DIR, METADATA_DIR, MASKED_IMAGES_DIR,
                    ATTENTION_MAPS_IMAGES_DIR, RAW_ATTENTION_MAPS_DIR, # Added raw maps dir
                    CHECKPOINT_DIR]
for type_dir in TYPE_DIRECTORIES:
    os.makedirs(type_dir, exist_ok=True)

# Warping parameters (use defaults from main_mmbench)
DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500
DEFAULT_EXP_SCALE = 1.0
DEFAULT_EXP_DIVISOR = 1.0
DEFAULT_APPLY_INVERSE = False
# --- End Configuration ---

# --- TextVQA Dataset Class (Copied & adapted from textvqa_processor.py) ---
class TextVQADataset(Dataset):

    def __init__(self, json_path, image_dir=None, download_images=False, transform=None):
        self.json_path = json_path
        self.image_dir = image_dir
        self.download_images = download_images
        self.transform = transform

        if download_images and image_dir:
            os.makedirs(image_dir, exist_ok=True)

        print(f"Loading TextVQA data from {json_path}...")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            self.metadata = {}
            self.samples = []
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_path}")
            self.metadata = {}
            self.samples = []
            return

        self.metadata = {
            'dataset_type': data.get('dataset_type'),
            'dataset_name': data.get('dataset_name'),
            'dataset_version': data.get('dataset_version')
        }
        self.samples = data.get('data', [])
        print(f"Loaded {len(self.samples)} samples from {self.metadata.get('dataset_name', 'Unknown')} {self.metadata.get('dataset_version', 'Unknown')}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.samples)}")

        sample = self.samples[idx].copy() # Return a copy to avoid modification issues

        # Get image (PIL Image or None)
        image = self._get_image(sample)

        # Apply transforms if specified
        if self.transform and image is not None:
            image = self.transform(image)

        # Add the loaded image to the sample dictionary for convenience
        sample['loaded_image'] = image

        return sample

    def _get_image(self, sample):
        """
        Get image for a sample, either from local storage or by downloading.
        Saves downloaded images to self.image_dir.
        """
        image_id = sample.get('image_id')
        if not image_id:
            print("Warning: Sample missing 'image_id'.")
            return None

        if self.image_dir:
            img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            if os.path.exists(img_path):
                try:
                    # print(f"Loading image from cache: {img_path}") # Optional: for debugging
                    return Image.open(img_path).convert('RGB')
                except Exception as e:
                    print(f"Warning: Error loading cached image {img_path}: {e}. Will attempt download if enabled.")

        if self.download_images:
            image_url = sample.get('flickr_original_url') or sample.get('flickr_300k_url')
            if image_url:
                # print(f"Downloading image for {image_id} from {image_url}") # Optional: for debugging
                try:
                    response = requests.get(image_url, timeout=15)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    if self.image_dir:
                        try:
                           img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
                           img.save(img_path)
                           # print(f"Image saved to cache: {img_path}") # Optional: for debugging
                        except Exception as e:
                           print(f"Warning: Error saving downloaded image to {img_path}: {e}")
                    return img
                except requests.exceptions.RequestException as e:
                    print(f"Warning: Failed to download image {image_id}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing downloaded image {image_id}: {e}")
            else:
                print(f"Warning: No valid image URL found for sample {image_id}.")
        elif not self.image_dir or not os.path.exists(os.path.join(self.image_dir, f"{image_id}.jpg")):
             print(f"Warning: Image not found locally and download is disabled: {image_id}")

        return None
# --- End TextVQA Dataset Class ---

# --- Helper Functions (Checkpointing, Signal Handling) ---
def save_checkpoint(data, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        # print(f"Checkpoint saved to {file_path}") # Optional: for debugging
    except Exception as e:
        print(f"Error saving checkpoint to {file_path}: {e}")

def load_checkpoint(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading checkpoint from {file_path}: {e}. Starting fresh.")
    return None

# --- Main Processing Logic ---
def main():

    dataset = TextVQADataset(
        json_path=TEXTVQA_JSON_PATH,
        image_dir=TEXTVQA_IMAGE_DIR,
        download_images=DOWNLOAD_IMAGES
    )

    # Prepare batch data (load images and questions)
    batch_images = []
    batch_questions = []
    batch_metadata_stubs = []
    skipped_count = 0

    print("Preparing batch (loading images and questions)...")
    for index in tqdm(range(len(dataset)), desc="Preparing samples"):
        try:
            entry = dataset[index] # __getitem__ loads/downloads image
            image = entry.get('loaded_image')
            question = entry.get('question')
            image_id = entry.get('image_id')

            if image is None:
                print(f"Warning: Image could not be loaded/downloaded for entry index {index}, image_id {image_id}. Skipping.")
                skipped_count += 1
                continue
                
            if not question:
                 print(f"Warning: Question is missing for entry index {index}, image_id {image_id}. Skipping.")
                 skipped_count += 1
                 continue

            batch_images.append(image)
            batch_questions.append(question)
            # Store necessary info (original sample dict excluding the large image object)
            metadata_stub = {k: v for k, v in entry.items() if k != 'loaded_image'}
            metadata_stub['original_index'] = index # Keep track of original dataset index
            batch_metadata_stubs.append(metadata_stub)

        except Exception as prep_err:
            print(f"Error preparing sample at index {index}: {prep_err}. Skipping.")
            skipped_count += 1
            continue
            
    num_to_process = len(batch_images)
    if num_to_process == 0:
        print("No valid entries found to process after preparation phase.")
        sys.exit(0)
        
    print(f"\nPrepared batch with {num_to_process} valid entries. {skipped_count} entries skipped during preparation.")

    # --- Checkpoint Loading ---
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "textvqa_processing_checkpoint.pkl")
    last_processed_internal_idx = -1
    processed_internal_indices = set()
    processed_count = 0
    failed_count = 0
    
    checkpoint_data = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        last_processed_internal_idx = checkpoint_data.get('last_processed_internal_idx', -1)
        processed_internal_indices = checkpoint_data.get('processed_internal_indices', set())
        processed_count = checkpoint_data.get('processed_count', 0)
        failed_count = checkpoint_data.get('failed_count', 0)
        print(f"Checkpoint loaded. Resuming after internal index {last_processed_internal_idx}. Already processed: {len(processed_internal_indices)}")
    else:
        print("No valid checkpoint found. Starting from the beginning.")
        
    # --- Signal Handling & Checkpoint Saving Setup ---
    global_checkpoint_data = {
        'last_processed_internal_idx': last_processed_internal_idx,
        'processed_internal_indices': processed_internal_indices,
        'processed_count': processed_count,
        'failed_count': failed_count
    }

    def save_current_checkpoint():
        save_checkpoint(global_checkpoint_data, checkpoint_file)
        print(f"\nCheckpoint saved. Last processed internal index: {global_checkpoint_data['last_processed_internal_idx']}")

    atexit.register(save_current_checkpoint)
    def signal_handler(sig, frame):
        print("\nInterruption detected. Saving progress before exiting...")
        save_current_checkpoint()
        print("Progress saved. Exiting.")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # --- End Checkpoint Setup ---

    # Filter out already processed items
    remaining_internal_indices = [i for i in range(num_to_process) if i not in processed_internal_indices]

    if not remaining_internal_indices:
        print("All items have already been processed according to checkpoint.")
        print(f"Total processed: {processed_count}")
        print(f"Total failed/skipped: {failed_count + skipped_count} ({failed_count} during processing, {skipped_count} during preparation)")
        return 0

    print(f"Processing {len(remaining_internal_indices)} remaining items out of {num_to_process} total prepared.")

    # --- Load LLaVA model once and set up hook logger ---
    try:
        tokenizer, model, image_processor, context_len, inner_model_name = get_model(MODEL_NAME)
        hl = hook_logger(model, model.device, layer_index=20)
    except Exception as load_err:
        print(f"Error loading LLaVA model or setting hook: {load_err}")
        return 1

    # --- Processing Loop (Chunking) ---
    chunk_size = 100 # Align with POPE processing chunk size
    for chunk_start in range(0, len(remaining_internal_indices), chunk_size):
        current_chunk_internal_indices = remaining_internal_indices[chunk_start : chunk_start + chunk_size]
        
        # Prepare inputs for the current chunk
        chunk_images = [batch_images[i] for i in current_chunk_internal_indices]
        chunk_questions = [batch_questions[i] for i in current_chunk_internal_indices]
        chunk_metadata_stubs = [batch_metadata_stubs[i] for i in current_chunk_internal_indices]

        print(f"\nProcessing chunk {chunk_start//chunk_size + 1}/{(len(remaining_internal_indices) + chunk_size - 1) // chunk_size}, internal indices {current_chunk_internal_indices[0]}-{current_chunk_internal_indices[-1]}")

        try:
            # **** Run attention extraction per item without using j ****
            masked_images = []
            attention_maps = []
            batch_mota_masks = []
            # Default visualization parameters (match attention_extraction defaults)
            enhance_coe = 10
            kernel_size = 3
            from PIL import Image as _PILImage  # avoid alias conflicts
            interpolate_method = getattr(_PILImage, "LANCZOS")
            grayscale = 0

            for img, question in tqdm(zip(chunk_images, chunk_questions), total=len(chunk_images), desc="Processing images"):
                # Reset hook logger for each new image/question
                hl.reinit()

                # Build arg bundle expected by getmask
                args = type('Args', (), {
                    "hl": hl,
                    "model_name": MODEL_NAME,
                    "model": model,
                    "tokenizer": tokenizer,
                    "image_processor": image_processor,
                    "context_len": context_len,
                    "query": question,
                    "conv_mode": None,
                    "image_file": img,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 20,
                })()
                mask_24x24, _out_text = getmask(args)  # torch.Tensor [24,24]
                # Save raw attention tensor in (1,1,24,24) format to match downstream handling
                attn_tensor = mask_24x24.clone().unsqueeze(0).unsqueeze(0)
                attention_maps.append(attn_tensor)
                merged_image, mota_mask = blend_mask(img, mask_24x24, enhance_coe, kernel_size, interpolate_method, grayscale)
                masked_images.append(merged_image)
                batch_mota_masks.append(mota_mask)

            # --- Validate API Output Sizes ---
            api_outputs_valid = True
            if batch_mota_masks is None or len(batch_mota_masks) != len(current_chunk_internal_indices):
                print(f"Error: API did not return expected number of MOTA masks. Expected {len(current_chunk_internal_indices)}, Got {len(batch_mota_masks) if batch_mota_masks is not None else 0}. Skipping chunk results.")
                batch_mota_masks = [None] * len(current_chunk_internal_indices) # Prevent index errors below
                api_outputs_valid = False # Mark as invalid to skip processing
            
            # Optional: Check other outputs similarly if they are critical
            if masked_images is None or len(masked_images) != len(current_chunk_internal_indices):
                 print(f"Warning: API did not return expected number of masked images. Expected {len(current_chunk_internal_indices)}, Got {len(masked_images) if masked_images is not None else 0}. Proceeding with Nones.")
                 masked_images = [None] * len(current_chunk_internal_indices)
                 
            if attention_maps is None or len(attention_maps) != len(current_chunk_internal_indices):
                 print(f"Warning: API did not return expected number of attention maps. Expected {len(current_chunk_internal_indices)}, Got {len(attention_maps) if attention_maps is not None else 0}. Proceeding with Nones.")
                 attention_maps = [None] * len(current_chunk_internal_indices)
            # --- End Validation ---

            if not api_outputs_valid:
                 print("Skipping result processing for this chunk due to invalid API output sizes.")
                 # Increment failed count for all items in this chunk
                 global_checkpoint_data['failed_count'] += len(current_chunk_internal_indices)
                 # Mark all items in chunk as processed (even though failed) to avoid retrying
                 for internal_idx in current_chunk_internal_indices:
                      global_checkpoint_data['processed_internal_indices'].add(internal_idx)
                      global_checkpoint_data['last_processed_internal_idx'] = max(global_checkpoint_data['last_processed_internal_idx'], internal_idx)
                 save_current_checkpoint() # Save checkpoint after failed chunk
                 continue # Move to the next chunk

            # --- Process and Save Results for the Chunk ---
            print(f"Saving results for {len(current_chunk_internal_indices)} items...")
            for chunk_idx, internal_idx in enumerate(tqdm(current_chunk_internal_indices, desc="Saving chunk results")):
                # Get corresponding data for this item
                original_image = batch_images[internal_idx]
                metadata_stub = chunk_metadata_stubs[chunk_idx] # Use chunk's metadata stub
                image_id = metadata_stub['image_id']
                original_dataset_index = metadata_stub['original_index'] # Get original index if needed
                
                # Get results from API batch
                mota_mask = batch_mota_masks[chunk_idx]
                masked_img = masked_images[chunk_idx]
                attention_map = attention_maps[chunk_idx]
                
                # Create a unique sample ID (no category needed)
                sample_id = f"{image_id}_{original_dataset_index}"
                
                # --- Define Save Paths (No Categories) ---
                original_save_path = os.path.join(ORIGINAL_IMAGES_DIR, f"{sample_id}_original.png")
                masked_image_path = os.path.join(MASKED_IMAGES_DIR, f"{sample_id}_masked.png")
                attention_map_img_path = os.path.join(ATTENTION_MAPS_IMAGES_DIR, f"{sample_id}_attn_map_img.png") # From API's attention_map
                attention_map_vis_path = os.path.join(ATTENTION_MAPS_DIR, f"{sample_id}_mota_mask_vis.png") # Visualization from mota_mask
                attention_map_npy_path = os.path.join(ATTENTION_MAPS_DIR, f"{sample_id}_mota_mask.npy") # Numpy from mota_mask
                # New path for the raw attention map tensor
                raw_attention_map_npy_path = os.path.join(RAW_ATTENTION_MAPS_DIR, f"{sample_id}_raw_attn.npy")
                metadata_path = os.path.join(METADATA_DIR, f"{sample_id}_metadata.json")
                warped_output_path = os.path.join(WARPED_IMAGES_DIR, f"{sample_id}_identity.png") # Identity warp
                # --- End Save Paths ---
                
                current_item_failed = False
                metadata = metadata_stub.copy() # Start with prepared metadata
                metadata["sample_id"] = sample_id
                metadata["api_model_name"] = MODEL_NAME
                # Add paths to metadata as they are successfully saved
                metadata["saved_paths"] = {
                    "original_image": None,
                    "masked_image": None,
                    "attention_map_image_from_api": None, # Visualization of raw map if tensor
                    "raw_attention_map_npy": None, # Saved raw tensor as npy
                    "mota_mask_visualization": None, # Visualization of processed mask
                    "mota_mask_npy": None, # Processed mask as npy
                    "warped_image_identity": None
                }

                try:
                    # 1. Save original image
                    try:
                        if isinstance(original_image, Image.Image):
                           original_image.save(original_save_path)
                           metadata["saved_paths"]["original_image"] = original_save_path
                        else: # Should not happen based on prep logic
                           print(f"Warning: Original image for {sample_id} is not a PIL Image.")
                    except Exception as img_err:
                        print(f"Warning: Could not save original image {original_save_path} for sample {sample_id}: {img_err}")
                        current_item_failed = True # Mark failure if original cannot be saved

                    # 2. Save masked image (if provided by API)
                    if isinstance(masked_img, Image.Image):
                        try:
                            masked_img.save(masked_image_path)
                            metadata["saved_paths"]["masked_image"] = masked_image_path
                        except Exception as masked_err:
                            print(f"Warning: Could not save masked image for sample {sample_id}: {masked_err}")
                    elif masked_img is not None:
                         print(f"Warning: masked_img for {sample_id} is not a PIL Image ({type(masked_img)}), not saving.")

                    # 3. Save attention map tensor as NPY and potentially as image
                    if torch.is_tensor(attention_map):
                         # 3a. Save raw tensor as NPY
                         try:
                             attn_np = attention_map.detach().cpu().numpy()
                             np.save(raw_attention_map_npy_path, attn_np)
                             metadata["saved_paths"]["raw_attention_map_npy"] = raw_attention_map_npy_path
                         except Exception as raw_npy_err:
                              print(f"Warning: Could not save raw attention tensor to NPY for {sample_id}: {raw_npy_err}")

                         # 3b. Save visualization of tensor as image
                         try:
                             attn_pil = T.ToPILImage()(attention_map.squeeze(0)) # Remove batch dim if present
                             attn_pil.save(attention_map_img_path)
                             metadata["saved_paths"]["attention_map_image_from_api"] = attention_map_img_path
                         except Exception as tensor_img_err:
                             print(f"Warning: Could not convert/save API attention tensor to image for {sample_id}: {tensor_img_err}")

                    elif isinstance(attention_map, Image.Image): # If API returns PIL for attention_map
                        try:
                            attention_map.save(attention_map_img_path)
                            metadata["saved_paths"]["attention_map_image_from_api"] = attention_map_img_path
                            # Optionally save as NPY too if needed, though less likely if it's already PIL
                            # np.save(raw_attention_map_npy_path, np.array(attention_map))
                            # metadata["saved_paths"]["raw_attention_map_npy"] = raw_attention_map_npy_path
                        except Exception as attn_img_err:
                            print(f"Warning: Could not save API attention map image for sample {sample_id}: {attn_img_err}")

                    elif attention_map is not None:
                         print(f"Warning: attention_map for {sample_id} is not PIL/Tensor ({type(attention_map)}), not saving.")
                         
                    # 4. Process MOTA mask (assuming it's the primary attention source for warping)
                    mota_mask_np = None # Initialize for warping step
                    if isinstance(mota_mask, Image.Image):
                        try:
                            # Save visualization (grayscale)
                            mota_mask_gray = mota_mask.convert('L') if mota_mask.mode != 'L' else mota_mask
                            mota_mask_gray.save(attention_map_vis_path)
                            metadata["saved_paths"]["mota_mask_visualization"] = attention_map_vis_path
                            
                            # Save numpy array
                            mota_mask_np = np.array(mota_mask_gray)
                            np.save(attention_map_npy_path, mota_mask_np)
                            metadata["saved_paths"]["mota_mask_npy"] = attention_map_npy_path
                            
                        except Exception as mota_err:
                            print(f"Warning: Error processing/saving MOTA mask for {sample_id}: {mota_err}")
                            current_item_failed = True # Fail if critical MOTA mask cannot be saved
                            mota_mask_np = None # Ensure it's None if saving failed
                    elif mota_mask is not None:
                        print(f"Warning: MOTA mask for {sample_id} is not a PIL Image ({type(mota_mask)}). Cannot save or warp.")
                        current_item_failed = True # Cannot proceed without valid mask
                    else: # mota_mask is None
                         print(f"Warning: MOTA mask is None for {sample_id}. Cannot save or warp.")
                         current_item_failed = True

                    # 5. Generate Warped Image (only if MOTA mask was processed successfully)
                    if mota_mask_np is not None and not current_item_failed:
                        try:
                            # Ensure we have the original image for warping
                            if isinstance(original_image, Image.Image):
                                success = save_warped_image(
                                    image_path=original_image,
                                    att_map=mota_mask_np,
                                    original_image_save_path=None,
                                    masked_overlay_save_path=None,
                                    output_path=warped_output_path,
                                    vis_path=None,
                                    width=DEFAULT_WIDTH,
                                    height=DEFAULT_HEIGHT,
                                    transform="identity",
                                    exp_scale=DEFAULT_EXP_SCALE,
                                    exp_divisor=DEFAULT_EXP_DIVISOR,
                                    apply_inverse=DEFAULT_APPLY_INVERSE
                                )
                                if success:
                                    metadata["saved_paths"]["warped_image_identity"] = warped_output_path
                                else:
                                    print(f"Warning: Failed to generate identity warped image for sample {sample_id}")
                                    # Don't mark as failed, maybe warping isn't critical
                            else:
                                print(f"Warning: Cannot warp {sample_id}, original image is not available.")
                        except Exception as warp_err:
                            print(f"Error during warping for sample {sample_id}: {warp_err}. Skipping warping.")
                            
                except Exception as item_proc_err:
                    print(f"Unexpected error processing item {sample_id}: {item_proc_err}")
                    current_item_failed = True
                finally:
                    try:
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                    except Exception as meta_err:
                        print(f"Critical Error: Could not save metadata for sample {sample_id}: {meta_err}")
                        current_item_failed = True # Metadata saving is crucial

                    if current_item_failed:
                        global_checkpoint_data['failed_count'] += 1
                    else:
                        global_checkpoint_data['processed_count'] += 1
                        
                    global_checkpoint_data['processed_internal_indices'].add(internal_idx)
                    global_checkpoint_data['last_processed_internal_idx'] = max(global_checkpoint_data['last_processed_internal_idx'], internal_idx)
                    save_checkpoint(global_checkpoint_data, checkpoint_file)

        except Exception as api_call_err:
            print(f"\nCritical Error during API call for chunk: {api_call_err}")
            print(f"Skipping remaining items in this chunk ({len(current_chunk_internal_indices)} items). Check API status.")
            global_checkpoint_data['failed_count'] += len(current_chunk_internal_indices)
            for internal_idx in current_chunk_internal_indices:
                global_checkpoint_data['processed_internal_indices'].add(internal_idx)
                global_checkpoint_data['last_processed_internal_idx'] = max(global_checkpoint_data['last_processed_internal_idx'], internal_idx)
            save_current_checkpoint() 
            continue

    print(f"\nProcessing finished.")
    final_processed = global_checkpoint_data['processed_count']
    final_failed = global_checkpoint_data['failed_count']
    print(f"Successfully processed: {final_processed}")
    print(f"Failed/Skipped: {final_failed + skipped_count} ({final_failed} during processing, {skipped_count} during preparation)")

    return 0 if final_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 