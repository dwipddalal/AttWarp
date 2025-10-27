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
from datasets import load_dataset
from io import BytesIO
import requests

# Import the working methods from new_method.py
from new_method import save_warped_image, llava_api

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Model name
MODEL_NAME = "llava-v1.5-7b"

# POPE Dataset Categories
CATEGORIES = ["adversarial", "popular", "random"]

# Default category to process if not specified
DEFAULT_CATEGORY = "adversarial"
# Special value to process all categories
ALL_CATEGORIES = "all"

# Define base output directory
OUTPUT_BASE_DIR = "/scratch/dwip2/CLIP/results/pope_processed"
ATTENTION_MAPS_DIR = os.path.join(OUTPUT_BASE_DIR, "attention_maps")
WARPED_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "warped_images")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, "visualizations")
ORIGINAL_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "original_images")
METADATA_DIR = os.path.join(OUTPUT_BASE_DIR, "metadata")
MASKED_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "masked_images")
ATTENTION_MAPS_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "attention_maps_images")
CHECKPOINT_DIR = os.path.join(OUTPUT_BASE_DIR, "checkpoints")

# Create base directory first
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create directories and category subdirectories
TYPE_DIRECTORIES = [ATTENTION_MAPS_DIR, WARPED_IMAGES_DIR, VISUALIZATIONS_DIR,
                    ORIGINAL_IMAGES_DIR, METADATA_DIR, MASKED_IMAGES_DIR, ATTENTION_MAPS_IMAGES_DIR]
for type_dir in TYPE_DIRECTORIES:
    os.makedirs(type_dir, exist_ok=True)
    for category in CATEGORIES:
        os.makedirs(os.path.join(type_dir, category), exist_ok=True)

# Warping parameters
DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500
DEFAULT_EXP_SCALE = 1.0
DEFAULT_EXP_DIVISOR = 1.0
DEFAULT_APPLY_INVERSE = False

def load_pope_dataset(split="test"):
    """
    Load the POPE dataset for a specific split.
    """
    try:
        dataset = load_dataset("lmms-lab/POPE")
        return dataset[split]
    except Exception as e:
        print(f"Error loading POPE dataset for split '{split}': {e}")
        raise e

def save_checkpoint(checkpoint_data, checkpoint_file):
    """Save processing checkpoint to resume later"""
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def process_category(category):
    """Process a single category of the POPE dataset"""
    print(f"Processing POPE dataset for category: {category}")
    
    # Load dataset
    try:
        dataset = load_pope_dataset(split="test")
        print(f"Loaded POPE dataset, split: test")
        print(f"Dataset size: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Create batch data structures
    batch_questions = []
    batch_metadata_stubs = []
    batch_pil_images = []  # Store actual PIL images
    skipped_count = 0
    
    print("Preparing batch...")
    
    # Process each example in the dataset
    for index, entry in enumerate(tqdm(dataset, desc=f"Finding images in POPE dataset")):
        # Check if entry belongs to the requested category
        entry_category = entry.get('category', None)
        if category != entry_category and category != ALL_CATEGORIES:
            continue
            
        question = entry.get('question', 'N/A')
        answer = entry.get('answer', 'N/A')
        question_id = entry.get('question_id', 'N/A')
        image_source = entry.get('image_source', 'Unknown')
        image = entry.get('image', None)
        
        # Skip if no image
        if image is None:
            skipped_count += 1
            continue
        
        # Store data for batch processing
        batch_pil_images.append(image)
        batch_questions.append(question)
        batch_metadata_stubs.append({
            "index": index,
            "category": entry_category,
            "question": question,
            "question_id": question_id,
            "answer": answer,
            "image_source": image_source
        })
    
    num_to_process = len(batch_pil_images)
    if num_to_process == 0:
        print(f"No valid entries found to process for category {category}.")
        return False
    
    print(f"\nPrepared batch with {num_to_process} valid entries. {skipped_count} entries skipped.")

    # Check for existing checkpoint file
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"pope_{category}_checkpoint.pkl")
    last_processed_index = -1
    processed_indices = set()
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                last_processed_index = checkpoint_data.get('last_processed_index', -1)
                processed_indices = checkpoint_data.get('processed_indices', set())
                processed_count = checkpoint_data.get('processed_count', 0)
                failed_count = checkpoint_data.get('failed_count', 0)
                
            print(f"Checkpoint found. Resuming from index {last_processed_index + 1}")
            print(f"Already processed {len(processed_indices)} items")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from beginning.")
            last_processed_index = -1
            processed_indices = set()
            processed_count = 0
            failed_count = 0
    else:
        processed_count = 0
        failed_count = 0
    
    # Function to save checkpoint
    def save_current_checkpoint():
        checkpoint_data = {
            'last_processed_index': last_processed_index,
            'processed_indices': processed_indices,
            'processed_count': processed_count,
            'failed_count': failed_count
        }
        save_checkpoint(checkpoint_data, checkpoint_file)
        print(f"\nCheckpoint saved for {category}. Last processed index: {last_processed_index}")
    
    # Register checkpoint saving on exit or interrupt
    atexit.register(save_current_checkpoint)
    
    # Handle interruption gracefully
    def signal_handler(sig, frame):
        print(f"\nInterruption detected while processing {category}. Saving progress before exiting...")
        save_current_checkpoint()
        print("Progress saved. Exiting.")
        sys.exit(0)
    
    # Register the signal handlers
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Only process items that weren't already processed
    remaining_indices = [i for i in range(num_to_process) if i not in processed_indices]
    
    if not remaining_indices:
        print(f"All items for category {category} have already been processed according to checkpoint.")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed/Skipped: {failed_count + skipped_count} ({failed_count} during processing, {skipped_count} pre-filtered)")
        return True
    
    print(f"Processing batch using model {MODEL_NAME}...")
    print(f"Will process {len(remaining_indices)} remaining items out of {num_to_process} total")
    
    # Process in smaller chunks to avoid losing too much work if interrupted
    chunk_size = 100
    
    for chunk_start in range(0, len(remaining_indices), chunk_size):
        chunk_indices = remaining_indices[chunk_start:chunk_start + chunk_size]
        chunk_images = [batch_pil_images[i] for i in chunk_indices]
        chunk_questions = [batch_questions[i] for i in chunk_indices]
        
        try:
            print(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(remaining_indices) + chunk_size - 1) // chunk_size}, indices {chunk_indices[0]}-{chunk_indices[-1]}")
            
            # Process current chunk with LLaVA API
            masked_images, attention_maps, batch_mota_masks = llava_api(
                chunk_images,  # Pass PIL images directly
                chunk_questions,
                model_name=MODEL_NAME
            )
            
            # Validate API outputs
            if batch_mota_masks is None or len(batch_mota_masks) != len(chunk_indices):
                raise ValueError(f"llava_api did not return the expected number of MOTA masks. Expected {len(chunk_indices)}, got {len(batch_mota_masks) if batch_mota_masks is not None else 0}")
            
            if masked_images is None or len(masked_images) != len(chunk_indices):
                print(f"Warning: llava_api did not return the expected number of masked images. Expected {len(chunk_indices)}, got {len(masked_images) if masked_images is not None else 0}")
                if masked_images is None:
                    masked_images = [None] * len(chunk_indices)
            
            if attention_maps is None or len(attention_maps) != len(chunk_indices):
                print(f"Warning: llava_api did not return the expected number of attention maps. Expected {len(chunk_indices)}, got {len(attention_maps) if attention_maps is not None else 0}")
                if attention_maps is None:
                    attention_maps = [None] * len(chunk_indices)
            
            print(f"Processing {len(chunk_indices)} items from current chunk...")
            for chunk_idx, idx in enumerate(tqdm(chunk_indices, desc=f"Saving results for {category}")):
                # Get all outputs for this item
                mota_mask = batch_mota_masks[chunk_idx] if chunk_idx < len(batch_mota_masks) else None
                masked_img = masked_images[chunk_idx] if chunk_idx < len(masked_images) else None
                attention_map = attention_maps[chunk_idx] if chunk_idx < len(attention_maps) else None
                
                # Get corresponding metadata stub and original image
                metadata_stub = batch_metadata_stubs[idx]
                original_image = batch_pil_images[idx]
                
                item_category = metadata_stub["category"]
                index = metadata_stub["index"]
                question_id = metadata_stub["question_id"]
                
                # Create a sample ID
                sample_id = f"{item_category}_{question_id}_{index}"
                
                # Define paths for this entry
                original_save_path = os.path.join(ORIGINAL_IMAGES_DIR, item_category, f"{sample_id}_original.png")
                masked_image_path = os.path.join(MASKED_IMAGES_DIR, item_category, f"{sample_id}_masked.png")
                attention_map_img_path = os.path.join(ATTENTION_MAPS_IMAGES_DIR, item_category, f"{sample_id}_attn_map.png")
                attention_map_np_path = os.path.join(ATTENTION_MAPS_DIR, item_category, f"{sample_id}_attention.npy")
                attention_image_path = os.path.join(ATTENTION_MAPS_DIR, item_category, f"{sample_id}_attention.png")
                metadata_path = os.path.join(METADATA_DIR, item_category, f"{sample_id}_metadata.json")
                warped_output_path = os.path.join(WARPED_IMAGES_DIR, item_category, f"{sample_id}_identity.png")
                
                current_item_failed = False
                
                # Initialize metadata with stub info
                metadata = {
                    "sample_id": sample_id,
                    "category": item_category,
                    "index": index,
                    "question": metadata_stub["question"],
                    "question_id": question_id,
                    "answer": metadata_stub["answer"],
                    "image_source": metadata_stub["image_source"],
                    "original_image": None,
                    "masked_image": None,
                    "attention_map_img": None,
                    "attention_map_png": None,
                    "attention_map_npy": None,
                    "warped_image_identity": None
                }
                
                try:
                    # Save original image
                    try:
                        original_image.save(original_save_path)
                        metadata["original_image"] = original_save_path
                    except Exception as img_err:
                        print(f"Warning: Could not save original image {original_save_path} for sample {sample_id}: {img_err}")
                    
                    # Save masked image if available
                    if isinstance(masked_img, Image.Image):
                        try:
                            masked_img.save(masked_image_path)
                            metadata["masked_image"] = masked_image_path
                        except Exception as masked_err:
                            print(f"Warning: Could not save masked image for sample {sample_id}: {masked_err}")
                    elif masked_img is not None:
                        print(f"Warning: masked_img is not a PIL Image for sample {sample_id}, it's a {type(masked_img)}")
                    
                    # Save attention map image if available
                    if isinstance(attention_map, Image.Image):
                        try:
                            attention_map.save(attention_map_img_path)
                            metadata["attention_map_img"] = attention_map_img_path
                        except Exception as attn_img_err:
                            print(f"Warning: Could not save attention map image for sample {sample_id}: {attn_img_err}")
                    elif attention_map is not None and torch.is_tensor(attention_map):
                        try:
                            # Try to save tensor as an image
                            attention_map_pil = T.ToPILImage()(attention_map.squeeze())
                            attention_map_pil.save(attention_map_img_path)
                            metadata["attention_map_img"] = attention_map_img_path
                        except Exception as tensor_err:
                            print(f"Warning: Could not convert attention tensor to image for sample {sample_id}: {tensor_err}")
                    elif attention_map is not None:
                        print(f"Warning: attention_map is not a PIL Image or tensor for sample {sample_id}, it's a {type(attention_map)}")
                    
                    # Process and save mota_mask
                    if isinstance(mota_mask, Image.Image):
                        # Save attention map from mota_mask
                        try:
                            mota_mask_gray = mota_mask.convert('L') if mota_mask.mode != 'L' else mota_mask
                            mota_mask_gray.save(attention_image_path)
                            mota_mask_np = np.array(mota_mask_gray)
                            np.save(attention_map_np_path, mota_mask_np)
                            metadata["attention_map_png"] = attention_image_path
                            metadata["attention_map_npy"] = attention_map_np_path
                            
                            # Generate warped image ONLY if attention map was saved successfully
                            try:
                                # Use the original image for warping
                                transform = "identity"  # Hardcode the transform type
                                
                                success = save_warped_image(
                                    original_image,
                                    mota_mask_np,
                                    warped_output_path,
                                    None,  # No visualization
                                    DEFAULT_WIDTH,
                                    DEFAULT_HEIGHT,
                                    transform,
                                    DEFAULT_EXP_SCALE,
                                    DEFAULT_EXP_DIVISOR,
                                    DEFAULT_APPLY_INVERSE
                                )
                                
                                if success:
                                    metadata["warped_image_identity"] = warped_output_path
                                else:
                                    print(f"Warning: Failed to generate identity warped image for sample {sample_id}")
                            except Exception as warp_err:
                                print(f"Error during warping for sample {sample_id}: {warp_err}. Skipping warping.")
                        except Exception as att_err:
                            print(f"Error processing/saving attention mask for {sample_id}: {att_err}. Skipping further processing for this item.")
                            current_item_failed = True
                    else:  # mota_mask is not a PIL Image
                        print(f"Warning: MOTA mask is not a PIL Image for sample {sample_id}, it's a {type(mota_mask)}. Skipping.")
                        if torch.is_tensor(mota_mask):
                            # Attempt cleanup if it's a tensor
                            try:
                                del mota_mask
                                torch.cuda.empty_cache()
                            except Exception as clean_err:
                                print(f"Warning: Error during tensor cleanup for sample {sample_id}: {clean_err}")
                        current_item_failed = True
                        
                except Exception as item_proc_err:
                    print(f"Unexpected error processing result for sample {sample_id}: {item_proc_err}")
                    current_item_failed = True
                    
                finally:
                    # Save metadata regardless of failures, as long as we have some data
                    saved_something = metadata["original_image"] or metadata["masked_image"] or metadata["attention_map_img"] or metadata["attention_map_png"]
                    if saved_something:
                        try:
                            with open(metadata_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2)
                        except Exception as meta_err:
                            print(f"Error saving metadata for sample {sample_id}: {meta_err}")
                            current_item_failed = True
                    
                    if current_item_failed and not saved_something:
                        failed_count += 1
                    else:
                        processed_count += 1
                    
                    # Mark this index as processed and update last_processed_index
                    processed_indices.add(idx)
                    last_processed_index = max(last_processed_index, idx)
                    
                    # Save checkpoint occasionally to ensure progress is not lost
                    if idx % 10 == 0:
                        save_current_checkpoint()
            
        except Exception as batch_proc_err:
            print(f"\nError during batch processing via llava_api for {category}: {batch_proc_err}")
            # Save checkpoint after exception before continuing
            save_current_checkpoint()
            # Continue with next chunk rather than exiting
            continue
    
    # Restore original signal handlers
    signal.signal(signal.SIGINT, original_sigint)
    signal.signal(signal.SIGTERM, original_sigterm)
    
    # Remove the atexit handler
    atexit.unregister(save_current_checkpoint)
    
    print(f"\nProcessing for category {category} completed.")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed/Skipped: {failed_count + skipped_count} ({failed_count} during processing, {skipped_count} pre-filtered)")
    
    return True

def main():
    # Get category from command line argument if provided, otherwise use default
    category_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CATEGORY
    
    # Track overall statistics
    total_success = 0
    total_failure = 0
    processed_categories = []
    failed_categories = []
    
    # Process either a single category or all categories
    if category_arg == ALL_CATEGORIES:
        print(f"Processing all {len(CATEGORIES)} categories from POPE dataset")
        
        # Add an overall progress tracker
        all_categories_progress_file = os.path.join(CHECKPOINT_DIR, "pope_all_categories_progress.json")
        completed_categories = set()
        
        # Check if we have an existing progress file
        if os.path.exists(all_categories_progress_file):
            try:
                with open(all_categories_progress_file, 'r') as f:
                    progress_data = json.load(f)
                    completed_categories = set(progress_data.get('completed_categories', []))
                print(f"Found progress file. Already completed: {', '.join(completed_categories) if completed_categories else 'None'}")
            except Exception as e:
                print(f"Error loading progress file: {e}. Starting from beginning.")
                completed_categories = set()
        
        # Function to save overall progress
        def save_overall_progress():
            try:
                with open(all_categories_progress_file, 'w') as f:
                    json.dump({
                        'completed_categories': list(completed_categories),
                        'processed_categories': processed_categories,
                        'failed_categories': failed_categories,
                        'total_success': total_success,
                        'total_failure': total_failure
                    }, f, indent=2)
            except Exception as e:
                print(f"Error saving overall progress: {e}")
        
        # Process each category in order
        for category in CATEGORIES:
            if category in completed_categories:
                print(f"Category {category} already processed. Skipping.")
                continue
            
            print(f"\n{'='*20} Processing category: {category} {'='*20}\n")
            success = process_category(category)
            
            if success:
                completed_categories.add(category)
                processed_categories.append(category)
                total_success += 1
            else:
                failed_categories.append(category)
                total_failure += 1
            
            # Save overall progress after each category
            save_overall_progress()
            
            # Release memory between categories
            torch.cuda.empty_cache()
        
        print(f"\n{'='*20} All Categories Processing Complete {'='*20}")
        print(f"Successfully processed categories: {', '.join(processed_categories) if processed_categories else 'None'}")
        print(f"Failed categories: {', '.join(failed_categories) if failed_categories else 'None'}")
    
    else:
        # Check if the category is valid
        valid_category = category_arg in CATEGORIES
        if not valid_category:
            print(f"Warning: Category '{category_arg}' not recognized. Using default: {DEFAULT_CATEGORY}")
            category_arg = DEFAULT_CATEGORY
        
        # Process the single category
        success = process_category(category_arg)
        
        # Return appropriate exit code
        if success:
            return 0
        else:
            return 1
    
    # Return appropriate exit code based on overall success
    return 0 if not failed_categories else 1

if __name__ == "__main__":
    sys.exit(main()) 