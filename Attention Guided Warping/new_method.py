import sys
import numpy as np
import cv2
import argparse
import os
from pathlib import Path
import pdb

import numpy as np
from PIL import Image
import pdb

import torch
import pdb

# Allow external override via environment variables
img_env = os.getenv("IMAGE_PATH")
q_env   = os.getenv("QUESTION_TEXT")

default_image = "/shared/nas2/dwip2/CLIP/images/image2.png"
default_query = "On the right desk, what is to the left of the laptop?"

images  = [img_env] if img_env else [default_image]
queries = [q_env] if q_env else [default_query]

def example_workflow():
    global images, mota_mask  # Ensure variables are accessible in main
    # If you are using the llava_api:
    from apiprompting import llava_api
    import torch
    # import pdb
    # 'images' and 'queries' are already set above from environment variables or defaults
    masked_images, attention_maps, mota_mask = llava_api(images, queries, model_name="llava-v1.5-7b")

    # save the attention maps as numpy arrays
    for i, att_map in enumerate(attention_maps):
        if isinstance(att_map, torch.Tensor):
            att_map = att_map.detach().cpu().numpy()
        np.save(f"/shared/nas2/dwip2/CLIP/attention_map_{i}.npy", att_map)
        # pdb.set_trace()
    # exit()
    # Fix: Handle the case where mota_mask is a list
    if isinstance(mota_mask, list):
        print(f"mota_mask is a list with {len(mota_mask)} elements")
        if mota_mask and len(mota_mask) > 0:
            # Use the first element of the list
            mota_mask = mota_mask[0]
            print(f"Using first element of mota_mask list, which is type: {type(mota_mask)}")
        else:
            print("mota_mask list is empty. Creating a default mask.")
            # Create a default mask if the list is empty
            default_shape = (500, 500)  # Adjust as needed
            mota_mask = Image.new('L', default_shape, 128)  # Create a gray mask

    # Convert PIL Image to grayscale and save
    if isinstance(mota_mask, Image.Image):
        # Convert to grayscale if it's not already
        if mota_mask.mode != 'L':
            mota_mask_gray = mota_mask.convert('L')
        else:
            mota_mask_gray = mota_mask
        
        # Save as grayscale image
        mota_mask_gray.save("/shared/nas2/dwip2/CLIP/mota_mask_gray.png")
        
        # Convert to numpy array and save
        mota_mask_np = np.array(mota_mask_gray)
        np.save("/shared/nas2/dwip2/CLIP/mota_mask.npy", mota_mask_np)
        print(f"Saved mota_mask as grayscale image and numpy file. Shape: {mota_mask_np.shape}")
        # pdb.set_trace()
    else:
        print(f"Warning: mota_mask is not a PIL Image, it's a {type(mota_mask)}")
        # Try to convert to numpy array if it's neither a list nor PIL Image
        try:
            mota_mask_np = np.array(mota_mask)
            print(f"Converted mota_mask to numpy array. Shape: {mota_mask_np.shape}")
            np.save("/shared/nas2/dwip2/CLIP/mota_mask.npy", mota_mask_np)
        except:
            print("Could not convert mota_mask to numpy array. Creating a default mask.")
            # Create a default mask
            mota_mask_np = np.ones((500, 500), dtype=np.uint8) * 128
            np.save("/shared/nas2/dwip2/CLIP/mota_mask.npy", mota_mask_np)

    # save the masked image
    if masked_images and len(masked_images) > 0:
        # PIL Image to NumPy array conversion for cv2
        masked_img = masked_images[0]  # Get the first image from the list
        if isinstance(masked_img, Image.Image):
            # Convert PIL Image to numpy array
            masked_img_np = np.array(masked_img)
            # Convert RGB to BGR for OpenCV
            if len(masked_img_np.shape) == 3 and masked_img_np.shape[2] == 3:
                masked_img_np = cv2.cvtColor(masked_img_np, cv2.COLOR_RGB2BGR)
            # Save using cv2
            cv2.imwrite("/shared/nas2/dwip2/CLIP/mota_mask_image.png", masked_img_np)
            print("Saved masked image to /shared/nas2/dwip2/CLIP/mota_mask_image.png")
        else:
            print(f"Warning: masked_images[0] is not a PIL Image, it's a {type(masked_img)}")
    else:
        print("No masked images to save")


# Define attention map transformation functions
def identity_transform(x):
    """No transformation, returns input as is."""
    return x

def identity_inverse(x):
    """Inverse of identity is identity."""
    return x

def square_transform(x):
    """Square the attention values."""
    return x**2

def square_inverse(x):
    """Inverse of square is square root."""
    return np.sqrt(np.maximum(x, 0))

def sqrt_transform(x):
    """Square root of attention values."""
    return np.sqrt(np.maximum(x, 0))

def sqrt_inverse(x):
    """Inverse of square root is square."""
    return x**2

# Configurable parameters for exponential transform
EXP_SCALE = 1.0  # Multiplier for input: exp(EXP_SCALE * x)
EXP_DIVISOR = 1.0  # Divisor for output: exp(x) / EXP_DIVISOR

# Apply inverse flag - enables "apply transform, take marginal, apply inverse" workflow
APPLY_INVERSE_TO_MARGINALS = False

def exp_transform(x):
    """Exponential of attention values with configurable scaling."""
    return np.exp(EXP_SCALE * x) / EXP_DIVISOR

def exp_inverse(x):
    """Inverse of exponential."""
    return np.log(np.maximum(x * EXP_DIVISOR, 1e-9)) / EXP_SCALE

def log_transform(x):
    """Log of attention values (with small epsilon to avoid log(0))."""
    return np.log(x + 1e-5)

def log_inverse(x):
    """Inverse of log is exp."""
    return np.exp(x) - 1e-5

# Mapping of transforms to their inverses
INVERSE_TRANSFORMS = {
    identity_transform: identity_inverse,
    square_transform: square_inverse,
    sqrt_transform: sqrt_inverse,
    exp_transform: exp_inverse,
    log_transform: log_inverse
}

# Choose the transformation function to apply
ATTENTION_TRANSFORM = sqrt_transform  # Default transform

# Constants 
EPSILON = 1e-9
BASE_ATTENTION = 1e-9 # Adjust visualization sensitivity

# --- Core Warping Logic ---
def warp_image_by_attention(image, att_map, new_width, new_height):
    """
    Warps an image based on attention map.
    Assumes image and att_map have the same HxW dimensions.
    """

    # Debug breakpoint removed for non-interactive runs

    h, w = image.shape[:2]
    att_map_float = att_map.astype(np.float64)
    att_map_float = np.maximum(att_map_float, 0)
    # Apply the selected transformation to attention map
    att_map_transformed = ATTENTION_TRANSFORM(att_map_float)
    
    att_map_biased = att_map_transformed + BASE_ATTENTION

    # Calculate Marginal Attention Profiles
    att_profile_x = np.sum(att_map_biased, axis=0) # Shape: (w,)
    att_profile_y = np.sum(att_map_biased, axis=1) # Shape: (h,)

    # Apply inverse function to marginal profiles if enabled
    if APPLY_INVERSE_TO_MARGINALS and ATTENTION_TRANSFORM in INVERSE_TRANSFORMS:
        inverse_func = INVERSE_TRANSFORMS[ATTENTION_TRANSFORM]
        # Remove BASE_ATTENTION before applying inverse
        att_profile_x = inverse_func(att_profile_x - BASE_ATTENTION * h) 
        att_profile_y = inverse_func(att_profile_y - BASE_ATTENTION * w)
        # Add back bias after inverse
        att_profile_x = att_profile_x + BASE_ATTENTION * h
        att_profile_y = att_profile_y + BASE_ATTENTION * w

    total_att_x = np.sum(att_profile_x)
    total_att_y = np.sum(att_profile_y)

    if total_att_x < EPSILON or total_att_y < EPSILON:
        print("Warning: Total attention is near zero.", file=sys.stderr)
        att_profile_x = np.ones(w, dtype=np.float64)
        att_profile_y = np.ones(h, dtype=np.float64)
        total_att_x = w * (np.mean(att_map_biased) * h) # Approximate total
        total_att_y = h * (np.mean(att_map_biased) * w) # Approximate total
        # Avoid division by zero later
        total_att_x = max(total_att_x, EPSILON)
        total_att_y = max(total_att_y, EPSILON)

    # Calculate Cumulative Profiles -> Forward Mapping
    cum_att_x = np.cumsum(att_profile_x)
    norm_cum_att_x = cum_att_x / total_att_x
    x_orig_coords = np.arange(w)
    x_new_map_fwd = np.concatenate(([0], norm_cum_att_x)) * new_width
    x_orig_map_fwd = np.concatenate(([0], x_orig_coords + 1))

    cum_att_y = np.cumsum(att_profile_y)
    norm_cum_att_y = cum_att_y / total_att_y
    y_orig_coords = np.arange(h)
    y_new_map_fwd = np.concatenate(([0], norm_cum_att_y)) * new_height
    y_orig_map_fwd = np.concatenate(([0], y_orig_coords + 1))

    x_new_map_fwd[-1] = new_width
    y_new_map_fwd[-1] = new_height

    # Inverse Mapping for cv2.remap
    x_target_coords = np.arange(new_width)
    y_target_coords = np.arange(new_height)
    map_x_orig = np.interp(x_target_coords, x_new_map_fwd, x_orig_map_fwd)
    map_y_orig = np.interp(y_target_coords, y_new_map_fwd, y_orig_map_fwd)

    final_map_x, final_map_y = np.meshgrid(map_x_orig, map_y_orig)
    final_map_x = final_map_x.astype(np.float32)
    final_map_y = final_map_y.astype(np.float32)

    # Apply Warp
    warped_image = cv2.remap(
        image, final_map_x, final_map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    # Ensure target shape (handle potential small deviations from remap)
    if warped_image.shape[0] != new_height or warped_image.shape[1] != new_width:
         final_channels = image.shape[2] if image.ndim == 3 else 1
         warped_image = cv2.resize(warped_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
         # Handle potential channel drop during resize
         if warped_image.ndim == 2 and final_channels == 3:
             warped_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
         elif warped_image.ndim == 3 and final_channels == 1 and image.ndim == 2:
             warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    return warped_image

def generate_visualization(image, att_map, warped_image, output_path, transform_name, attention_alpha=0.5):
    """Generate visualization with original image, attention map, and warped result"""
    if image is None or att_map is None or warped_image is None:
        print("Cannot generate visualization: missing data")
        return

    # Normalize attention map for visualization
    att_map_norm = att_map.copy()
    min_val, max_val = np.min(att_map), np.max(att_map)
    if max_val > min_val + EPSILON:
        att_map_norm = (att_map_norm - min_val) / (max_val - min_val)
    else:
        att_map_norm = np.zeros_like(att_map)
    
    # Convert to heatmap visualization for overlay
    att_map_color = cv2.applyColorMap((att_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Create the overlay image
    overlay = image.copy()
    cv2.addWeighted(att_map_color, attention_alpha, image, 1 - attention_alpha, 0, overlay)
    
    # Create a combined visualization
    h, w = image.shape[:2]
    h_warped, w_warped = warped_image.shape[:2]
    
    # Make all images the same height for visualization
    target_height = max(h, h_warped)
    
    # Resize if needed
    if h != target_height:
        scale = target_height / h
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, target_height))
        overlay = cv2.resize(overlay, (new_w, target_height))
        w = new_w
    
    if h_warped != target_height:
        scale = target_height / h_warped
        new_w_warped = int(w_warped * scale)
        warped_image = cv2.resize(warped_image, (new_w_warped, target_height))
        w_warped = new_w_warped
    
    # Create visualization with original + attention + warped
    visualization = np.zeros((target_height, w + w + w_warped, 3), dtype=np.uint8)
    visualization[:, :w] = image
    visualization[:, w:2*w] = overlay
    visualization[:, 2*w:] = warped_image
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualization, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, "Attention Map", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, f"Warped ({transform_name})", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Add separator lines
    cv2.line(visualization, (w, 0), (w, target_height), (255, 255, 255), 2)
    cv2.line(visualization, (2*w, 0), (2*w, target_height), (255, 255, 255), 2)

    # Add grid to warped image
    grid_spacing = 20
    for x in range(2*w, 2*w + w_warped, grid_spacing):
        cv2.line(visualization, (x, 0), (x, target_height), (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, target_height, grid_spacing):
        cv2.line(visualization, (2*w, y), (2*w + w_warped, y), (255, 255, 255), 1, cv2.LINE_AA)
    
    # Save visualization
    cv2.imwrite(output_path, visualization)
    
    print(f"Visualization saved to {output_path}")

def resize_image_to_match_attmap(image, att_map):
    """Resizes image to match attention map dimensions if necessary."""
    if image is None or att_map is None:
        return None

    target_h, target_w = att_map.shape[:2]
    current_h, current_w = image.shape[:2]

    if (current_h, current_w) == (target_h, target_w):
        print("Image and attention map dimensions match.")
        return image.copy()
    else:
        print(f"Resizing image from {image.shape[:2]} to match attention map {att_map.shape[:2]}")
        try:
            resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if resized_image.shape[:2] != (target_h, target_w):
                raise RuntimeError("Resize resulted in unexpected shape.")
            print("Image resized successfully.")
            return resized_image
        except Exception as e:
            print(f"Error resizing image: {e}")
            return None

def set_transform_function(transform_name, exp_scale=1.0, exp_divisor=1.0, apply_inverse=False):
    """Sets the global transformation function based on name."""
    global ATTENTION_TRANSFORM, EXP_SCALE, EXP_DIVISOR, APPLY_INVERSE_TO_MARGINALS
    
    # Set exponential parameters if provided
    EXP_SCALE = exp_scale
    EXP_DIVISOR = exp_divisor
    APPLY_INVERSE_TO_MARGINALS = apply_inverse
    
    # Set the transformation function
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
        print(f"Unknown transform: {transform_name}. Using identity transform.")
        ATTENTION_TRANSFORM = identity_transform
        return "identity"
    
    return transform_name

def save_warped_image(image_path, att_map, 
                      original_image_save_path,  # New path for resized original
                      masked_overlay_save_path, # New path for masked overlay
                      output_path, # Path for warped image
                      vis_path=None, 
                      width=500, height=500, transform="identity", # width/height now act as defaults/guide for viz
                      exp_scale=1.0, exp_divisor=1.0, apply_inverse=False, attention_alpha=0.5):
    """Process and save warped image, original, and masked overlay, all in original input image dimensions."""
    try:
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        else:
            # If image_path is already a PIL image
            image = np.array(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        
        input_original_h, input_original_w = image.shape[:2]

        # Save a copy of the original image (no resize needed here for this specific output)
        if original_image_save_path:
            cv2.imwrite(original_image_save_path, image.copy())
            print(f"Original image saved to {original_image_save_path} with shape {image.shape[:2]}")

        # Handle attention map
        if isinstance(att_map, np.ndarray):
            pass
        elif isinstance(att_map, Image.Image):
            att_map = np.array(att_map)
        elif isinstance(att_map, list):
            if len(att_map) > 0:
                first_element = att_map[0]
                if isinstance(first_element, np.ndarray):
                    att_map = first_element
                elif isinstance(first_element, Image.Image):
                    att_map = np.array(first_element)
                else:
                    att_map = np.array(first_element)
            else:
                # Use provided width/height for default att_map if list is empty
                att_map = np.ones((height, width), dtype=np.float32) * 128 
            
        if att_map.ndim == 3:
            att_map = np.mean(att_map, axis=2)
        elif att_map.ndim != 2:
            raise ValueError(f"Attention map must be 2D, got shape {att_map.shape}")
        
        # Create and save masked overlay image (using original input dimensions)
        if masked_overlay_save_path:
            # Base image for overlay is the original image itself
            overlay_base_image = image.copy()
            if overlay_base_image.ndim == 2: # if grayscale
                overlay_base_image = cv2.cvtColor(overlay_base_image, cv2.COLOR_GRAY2BGR)

            # Resize attention map to original input dimensions for overlay
            att_map_resized_for_overlay = cv2.resize(att_map.copy(), (input_original_w, input_original_h), interpolation=cv2.INTER_LINEAR)
            
            att_map_norm_overlay = att_map_resized_for_overlay.copy()
            min_val_ov, max_val_ov = np.min(att_map_norm_overlay), np.max(att_map_norm_overlay)
            if max_val_ov > min_val_ov + EPSILON:
                att_map_norm_overlay = (att_map_norm_overlay - min_val_ov) / (max_val_ov - min_val_ov)
            else:
                att_map_norm_overlay = np.zeros_like(att_map_norm_overlay)
            
            att_map_color_overlay = cv2.applyColorMap((att_map_norm_overlay * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            masked_overlay_image = cv2.addWeighted(att_map_color_overlay, attention_alpha, overlay_base_image, 1 - attention_alpha, 0)
            cv2.imwrite(masked_overlay_save_path, masked_overlay_image)
            print(f"Masked overlay image saved to {masked_overlay_save_path} with shape {masked_overlay_image.shape[:2]}")

        # Image for warping: resize original image to match attention map dimensions (as expected by warp_image_by_attention)
        image_for_warping = resize_image_to_match_attmap(image, att_map)
        if image_for_warping is None:
            raise ValueError("Failed to resize image to match attention map dimensions for warping")
        
        # Set transform function
        transform_name = set_transform_function(transform, exp_scale, exp_divisor, apply_inverse)
        
        # Process image (Warping) - output will be of input_original_w, input_original_h
        warped_image = warp_image_by_attention(image_for_warping, att_map, input_original_w, input_original_h)
        if warped_image is None:
            raise ValueError("Warping failed")
        
        # Save warped image (now in original input dimensions)
        cv2.imwrite(output_path, warped_image)
        print(f"Warped image saved to {output_path} with shape {warped_image.shape[:2]}")
        
        # Generate and save visualization strip if requested
        # Inputs to generate_visualization are: 
        # image_for_warping (original image resized to att_map dims), 
        # att_map (original resolution att_map), 
        # warped_image (warped to original input image dims)
        if vis_path:
            generate_visualization(image_for_warping, att_map, warped_image, vis_path, transform_name, attention_alpha)
        
        return True
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return False

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Attention-Based Non-Uniform Image Warping")
    
    # Required arguments
    parser.add_argument("--image", required=False, default="/shared/nas2/dwip2/CLIP/images/image2.png", help="Path to input image file")
    parser.add_argument("--attention-map", required=False, default="/shared/nas2/dwip2/CLIP/mota_mask.npy", help="Path to attention map .npy file")
    # Output base filename for warped image, directory will be auto-generated
    parser.add_argument("--output", required=False, default="mota_warped.png", help="Base filename for warped output image") 
    
    # Optional arguments
    # Visualization base filename, directory will be auto-generated
    parser.add_argument("--visualization", default=None, help="Base filename for visualization with input, attention map, and output") 
    parser.add_argument("--width", type=int, default=500, help="Target width for warped image")
    parser.add_argument("--height", type=int, default=500, help="Target height for warped image")
    parser.add_argument("--transform", choices=["identity", "square", "sqrt", "exp", "log"], 
                        default="identity", help="Attention transformation function")
    parser.add_argument("--exp-scale", type=float, default=1.0, help="Scale for exponential transform")
    parser.add_argument("--exp-divisor", type=float, default=1.0, help="Divisor for exponential transform")
    parser.add_argument("--apply-inverse", action="store_true", 
                        help="Apply inverse transform to marginal profiles")
    parser.add_argument("--attention-alpha", type=float, default=0.4, 
                        help="Alpha blending value for attention map overlay (0.0-1.0)")
    
    args = parser.parse_args()

    # --- Create unique output directory for this run ---
    base_output_dir = "/shared/nas2/dwip2/CLIP/output_runs"
    os.makedirs(base_output_dir, exist_ok=True)
    
    run_id = 0
    while True:
        current_run_dir = os.path.join(base_output_dir, f"run_{run_id}")
        if not os.path.exists(current_run_dir):
            os.makedirs(current_run_dir)
            break
        run_id += 1
    print(f"Saving outputs to: {current_run_dir}")

    # Define full paths for all outputs within the run-specific directory
    original_image_save_path = os.path.join(current_run_dir, "original_image.png") # Renamed
    masked_overlay_save_path = os.path.join(current_run_dir, "masked_overlay_image.png") # Renamed
    warped_image_save_path = os.path.join(current_run_dir, os.path.basename(args.output))
    
    visualization_save_path = None
    if args.visualization:
        visualization_save_path = os.path.join(current_run_dir, os.path.basename(args.visualization))
    
    # Handle mota_mask being a list for the main function
    mota_mask_to_use = mota_mask
    if isinstance(mota_mask, list):
        if mota_mask and len(mota_mask) > 0:
            print(f"Main: mota_mask is a list, using first element of {len(mota_mask)} items")
            mota_mask_to_use = mota_mask[0]
        else:
            print("Main: mota_mask is an empty list, creating default mask")
            mota_mask_to_use = np.ones((args.height, args.width), dtype=np.float32) * 128
    
    # Process image
    success = save_warped_image(
        image_path=images[0], 
        att_map=mota_mask_to_use,  # Use the properly handled mask
        original_image_save_path=original_image_save_path,
        masked_overlay_save_path=masked_overlay_save_path,
        output_path=warped_image_save_path,
        vis_path=visualization_save_path,
        width=args.width, 
        height=args.height, 
        transform=args.transform,
        exp_scale=args.exp_scale,
        exp_divisor=args.exp_divisor,
        apply_inverse=args.apply_inverse,
        attention_alpha=args.attention_alpha
    )
    
    return 0 if success else 1

if __name__ == '__main__':
    example_workflow()
    sys.exit(main())