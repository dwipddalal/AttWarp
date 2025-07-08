import os
import sys
import json
import glob
import time
import argparse
from typing import cast, List

import torch
from PIL import Image
from tqdm import tqdm

# LLaVA imports
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    IMAGE_TOKEN_INDEX,
)

# -----------------------------------------------------------------------------
# Path configuration (UPDATE HERE IF DIRECTORY CHANGES)
# -----------------------------------------------------------------------------
RESULTS_DIR = "/shared/nas2/dwip2/CLIP/results/docvqa_processed"
METADATA_DIR = os.path.join(RESULTS_DIR, "metadata")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "accuracy_results_warped")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def load_llava_model(model_path: str = "liuhaotian/llava-v1.5-7b"):
    """Load the tokenizer, model, and image processor from LLaVA."""
    disable_torch_init()
    print(f"Loading LLaVA model: {model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
    )
    return tokenizer, model, image_processor


def get_llava_response(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    question: str,
) -> str | None:
    """Run LLaVA on a single (image, question) pair and return the raw answer text."""
    model_name_str = model.config._name_or_path.lower()
    if "llama-2" in model_name_str:
        conv_mode = "llava_llama_2"
    elif "v1.6" in model_name_str or "v1.5" in model_name_str:
        conv_mode = "llava_v1"
    elif "mpt" in model_name_str:
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # Build prompt
    if model.config.mm_use_im_start_end:
        prompt = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        )
    else:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
    prompt += " Answer briefly in a word or short phrase."

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=torch.float16
        )
        input_ids = cast(
            torch.Tensor,
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ),
        ).unsqueeze(0).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                temperature=0,
                max_new_tokens=64,
                use_cache=True,
            )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def is_substring_match(predicted: str | None, ground_truths: List[str]) -> bool:
    """Return True iff predicted answer is substring of any GT or vice-versa (case-insensitive)."""
    if not predicted or not ground_truths:
        return False
    predicted = predicted.strip().lower()
    for gt in ground_truths:
        gt = gt.strip().lower()
        if predicted in gt or gt in predicted:
            return True
    return False

# -----------------------------------------------------------------------------
# Main evaluation routine
# -----------------------------------------------------------------------------

def evaluate_docvqa_accuracy(model_path: str = "liuhaotian/llava-v1.5-7b"):
    """Evaluate LLaVA on DocVQA warped images using substring-match accuracy."""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    moving_log = os.path.join(OUTPUT_DIR, f"docvqa_moving_accuracy_{timestamp}.csv")
    detailed_log = os.path.join(OUTPUT_DIR, f"docvqa_detailed_log_{timestamp}.txt")

    # Initialise detailed log
    with open(detailed_log, "w", encoding="utf-8") as f:
        f.write("DOCVQA FULL EVALUATION ‑ DETAILED LOG\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Evaluation Metric: Substring match (warped images)\n")
        f.write("=" * 80 + "\n\n")

    # Load LLaVA
    tokenizer, model, image_processor = load_llava_model(model_path)

    # Aggregates (track warped images only)
    stats = {
        "warped": {"correct": 0, "total": 0},
    }
    detailed_results: List[dict] = []
    sample_idx = 0

    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    if not metadata_files:
        print(f"No metadata JSON files found in {METADATA_DIR}")
        sys.exit(1)

    print(f"Found {len(metadata_files)} metadata files – beginning evaluation…")

    for meta_path in tqdm(metadata_files, desc="Evaluating samples"):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Could not read {meta_path}: {e}")
            continue

        sample_id = meta.get("sample_id")
        question = meta.get("question")
        answers = meta.get("answers", [])
        paths = meta.get("saved_paths", {})
        # Only need the warped image path for evaluation
        warp_path = paths.get("warped_image_identity")

        if not (sample_id and question and answers and warp_path):
            continue
        if not os.path.exists(warp_path):
            continue

        # Inference (warped image only)
        warp_ans_raw = get_llava_response(model, tokenizer, image_processor, warp_path, question)
        if warp_ans_raw is None:
            continue

        # Accuracy (warped only)
        warp_correct = is_substring_match(warp_ans_raw, answers)

        # Update stats
        stats["warped"]["total"] += 1
        stats["warped"]["correct"] += int(warp_correct)

        # Detailed storage
        detailed_results.append(
            {
                "sample_id": sample_id,
                "question": question,
                "ground_truth_answers": answers,
                "warped_answer": warp_ans_raw,
                "warped_correct": warp_correct,
            }
        )

        # Append to log
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"SAMPLE: {sample_id} ‑ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n")
            f.write(f"QUESTION: {question}\n\n")
            f.write(f"GROUND TRUTH ANSWERS: {', '.join(answers)}\n\n")
            f.write(f"WARPED ANSWER:   {warp_ans_raw} (Correct: {warp_correct})\n")

        # Moving-average log every 10 samples
        sample_idx += 1
        if sample_idx % 10 == 0:
            warp_acc = stats["warped"]["correct"] / max(1, stats["warped"]["total"])
            timestamp_now = time.strftime("%Y-%m-%d %H:%M:%S")
            file_exists = os.path.exists(moving_log)
            with open(moving_log, "a") as f:
                if not file_exists:
                    f.write("Timestamp,Iteration,Total_Samples,Warped_Accuracy\n")
                f.write(
                    f"{timestamp_now},{sample_idx},{stats['warped']['total']},{warp_acc:.4f}\n"
                )
            print(f"\n[Stats @ {sample_idx}] Warped accuracy: {warp_acc:.4f}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total = stats["warped"]["total"]
    if total == 0:
        print("No samples processed successfully.")
        return None

    final_warp_acc = stats["warped"]["correct"] / total

    results = {
        "model": model_path,
        "dataset": "DocVQA",
        "evaluation_metric": "Substring match (warped images only)",
        "overall_warped_accuracy": final_warp_acc,
        "total_samples_evaluated": total,
        "detailed_results": detailed_results,
        "log_files": {
            "moving_average": moving_log,
            "detailed": detailed_log,
        },
    }

    results_path = os.path.join(OUTPUT_DIR, f"docvqa_accuracy_{timestamp}.json")
    summary_path = os.path.join(OUTPUT_DIR, f"docvqa_accuracy_{timestamp}_summary.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DocVQA Accuracy Evaluation with LLaVA (Warped Images)\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Total samples evaluated: {total}\n\n")
        f.write(f"Warped accuracy:   {final_warp_acc:.4f} ({stats['warped']['correct']}/{total})\n")

    print("\nEvaluation complete!")
    print(f"  Warped accuracy:   {final_warp_acc:.4f}")
    print(f"Results JSON saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")

    return summary_path, moving_log, detailed_log


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLaVA accuracy on DocVQA dataset (warped images)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
        help="LLaVA model checkpoint/path",
    )
    args = parser.parse_args()

    evaluate_docvqa_accuracy(args.model) 