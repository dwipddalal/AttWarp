# Attention Guided Warping

This folder contains the attention-guided warping pipeline that is intended to be used alongside an official MLLM (multimodal LLM) repository.
The implementation assumes you have the model code installed (for example, the official **LLaVA** repo) and uses its model/tokenizer utilities to
extract attention maps, generate masks, and warp images.

## Repository dependency (MLLM)
- **Expected usage:** run these scripts inside an environment where the official MLLM repo is installed and importable.
- **Example:** LLaVA (e.g., `liuhaotian/llava-v1.5-7b`) provides the model loading, tokenizer, and image processing utilities used in this code.

## File overview and key functions

### `main.py`
Runs the TextVQA processing pipeline end-to-end: loads the dataset, calls the LLaVA attention API, saves masks/attention outputs, and produces warped images.

Key components:
- **`TextVQADataset`**: PyTorch dataset that loads TextVQA JSON, fetches images, and yields samples.
- **`save_checkpoint(data, file_path)`**: writes progress checkpoint to disk.
- **`load_checkpoint(file_path)`**: loads checkpoint if it exists.
- **`main()`**: orchestrates dataset preparation, batch processing, API calls, file saving, and checkpointing.

### `new_method.py`
Implements attention-map transforms and the warping pipeline, plus a CLI for running warps locally.

Key components:
- **`example_workflow()`**: sample end-to-end flow using `llava_api` to extract attention and save results.
- **Transform helpers**: `identity_transform`, `square_transform`, `sqrt_transform`, `exp_transform`, `log_transform` and their inverses.
- **`warp_image_by_attention(image, att_map, new_width, new_height)`**: computes non-uniform warping using an attention map.
- **`generate_visualization(...)`**: creates a side-by-side visualization (original, attention overlay, warped).
- **`resize_image_to_match_attmap(image, att_map)`**: ensures the image matches attention-map resolution.
- **`set_transform_function(transform_name, exp_scale, exp_divisor, apply_inverse)`**: configures the global attention transform.
- **`save_warped_image(...)`**: saves original, overlay, and warped output images.
- **`main()`**: CLI entrypoint for running a warp on a single image + attention map.

### `attention_extraction/functions.py`
Utility functions for loading images and generating attention masks via LLaVA.

Key components:
- **`image_parser(args)`**: splits image path list for multi-image prompts.
- **`load_image(image_file)` / `load_images(image_files)`**: loads images from local path or URL.
- **`getmask(args)`**: runs LLaVA generation to return a 24×24 attention mask and model output.
- **`get_model(model_path)`**: loads LLaVA tokenizer/model/image processor.

### `attention_extraction/hook.py`
Defines a lightweight hook manager used to intercept model internals.

Key components:
- **`HookManager.register(name, func)` / `unregister(name, func)`**: manage hook callbacks.
- **`HookManager.__call__(name, **kwargs)`**: execute callbacks for a hook name.
- **`HookManager.fork(name)` / `fork_iterative(name, iteration)`**: create scoped hook managers.
- **`HookManager.finalize()`**: validates that registered hooks were used.

### `attention_extraction/llava.py`
LLaVA-specific attention extraction and mask blending utilities.

Key components:
- **`MaskHookLogger`**: collects attention and projection statistics from model layers.
- **`hook_logger(model, device, layer_index)`**: installs hooks on LLaVA layers.
- **`blend_mask(...)`**: normalizes a mask, resizes it, and overlays it on the image.
- **`llava_api(images, queries, model_name, ...)`**: main API that returns masked images, attention maps, and MOTA masks.

### `evaluate_accuracy.py`
Evaluates warped-image performance on TextVQA with LLaVA.

Key components:
- **Text preprocessing**: `processPunctuation`, `processDigitArticle`, `process_text`.
- **`calculate_vqa_accuracy(predicted_answer, ground_truth_answers, threshold)`**: VQA-style accuracy check.
- **`load_llava_model(model_path)`**: loads LLaVA model for evaluation.
- **`get_llava_response(model, tokenizer, image_processor, image_path, question)`**: runs inference on an image + question.
- **`evaluate_textvqa_accuracy(model_path)`**: end-to-end evaluation over saved metadata and warped images.

## Notes
- The attention extraction logic expects **LLaVA** or another MLLM repo to be installed and importable.
- Paths like `results/` and model identifiers can be customized to match your environment.
