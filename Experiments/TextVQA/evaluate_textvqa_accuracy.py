import os
import sys
import json
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter # Use Counter for efficient counting
from typing import cast  # For static type casting
import argparse
import time
import re  # Add regex import for text preprocessing

# Import LLaVA model loading utilities (ensure LLaVA environment is active)
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, IMAGE_TOKEN_INDEX

# Define constants for TextVQA results
# --- Path constants: replace absolute paths with relative ---

# BASE_DIR = os.path.join(os.path.dirname(__file__), "results", "textvqa")
BASE_DIR = "/shared/nas2/dwip2/CLIP/results/textvqa_processed_full"
WARPED_IMAGES_DIR = os.path.join(BASE_DIR, "warped_images")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
OUTPUT_DIR = os.path.join(BASE_DIR, "accuracy_results_llava")

# --- VQA Text Preprocessing Functions ---
contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap    = { 'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
articles     = ['a', 'an', 'the']

periodStrip  = re.compile("(?<!\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
punct        = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

def processPunctuation(inText):
    outText = inText
    # First handle periods separately to ensure consistent behavior with text like "no.1"
    outText = periodStrip.sub(" ", outText, re.UNICODE)  # Replace periods not between digits with spaces
    
    # Then handle other punctuation
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    
    # Normalize multiple spaces to single space
    outText = re.sub(r'\s+', ' ', outText).strip()
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def process_text(text):
    """
    Process text by removing punctuation, articles, and handling contractions.
    Includes special case handling for common patterns like "No.1" and "No 1".
    """
    # Special case handling for patterns like "No.1" / "No 1"
    text = text.lower()  # Convert to lowercase first
    text = re.sub(r'no\s*\.\s*(\d+)', r'no \1', text)  # no.1 -> no 1
    text = re.sub(r'no\s+(\d+)', r'no \1', text)       # no 1 -> no 1 (normalized spacing)
    text = re.sub(r'number\s*\.\s*(\d+)', r'number \1', text)  # number.1 -> number 1
    text = re.sub(r'number\s+(\d+)', r'number \1', text)       # number 1 -> number 1
    
    # Regular processing
    text = processPunctuation(text)
    text = processDigitArticle(text)
    return text

def get_acc(pred, gts):
    pred = process_text(pred)
    gts = [process_text(gt) for gt in gts]
    same_num = sum([1 if pred == gt else 0 for gt in gts])
    return 100*min(0.33333*same_num, 1)
# --- End VQA Text Preprocessing ---

def load_llava_model(model_path="liuhaotian/llava-v1.5-7b"):
    disable_torch_init()
    print(f"Loading LLaVA model: {model_path}")
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )
        return tokenizer, model, image_processor, context_len, get_model_name_from_path(model_path)
    except Exception as e:
        print(f"Fatal Error loading LLaVA model: {e}")
        print("Please ensure the LLaVA repository is correctly set up and the model path is valid.")
        sys.exit(1)

def get_llava_response(model, tokenizer, image_processor, image_path, question):
    """ Gets LLaVA response for a given image and plain question. Returns the raw output string. """
    formatted_question = question # TextVQA uses plain questions
    model_name_str = model.config._name_or_path.lower()
    if "llama-2" in model_name_str:
        conv_mode = "llava_llama_2"
    elif "v1.6" in model_name_str or "v1.5" in model_name_str:
        conv_mode = "llava_v1"
    elif "mpt" in model_name_str:
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + formatted_question
    else:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + formatted_question
    conv = conv_templates[conv_mode].copy()
    prompt =  prompt + "Answer in a single word or key phrase."
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"Prompt: {prompt}")
    
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=torch.float16
        )
        # Cast to torch.Tensor to satisfy static type checkers
        input_ids = cast(torch.Tensor, tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        )).unsqueeze(0).to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                temperature=0,
                max_new_tokens=64, 
                use_cache=True
            )

        # Decode the raw output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Return the raw, unparsed, uncleaned output string
        return outputs

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def update_moving_accuracy_log(log_file, overall_results, iteration):
    """Update the moving average accuracy log file (overall only)."""
    try:
        warp_total = overall_results['warped']['total']
        warp_correct = overall_results['warped']['correct']

        warp_acc = warp_correct / max(1, warp_total)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        file_exists = os.path.exists(log_file)
        with open(log_file, 'a') as f:
            if not file_exists:
                f.write("Timestamp,Iteration,Total_Samples,Warped_Accuracy\n")
            f.write(f"{timestamp},{iteration},{warp_total},{warp_acc:.4f}\n")
        
        # Print current stats to console
        print(f"\n--- Moving Average (Sample {iteration}, Total: {warp_total}) ---")
        print(f"  Warped accuracy:   {warp_acc:.4f} ({warp_correct}/{warp_total})")
    except Exception as e:
        print(f"Error updating moving average log: {e}")

def update_detailed_log(log_file, sample_data, overall_results):
    """Update the detailed log file with sample-by-sample information."""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        sample_id = sample_data['sample_id']
        question = sample_data['question']
        ground_truth_answers = sample_data['ground_truth_answers'] # List of answers
        warped_answer = sample_data['warped_answer']
        warped_correct = sample_data['warped_correct']   # Based on VQA Acc check
        
        # Calculate current overall accuracies
        warp_total = overall_results['warped']['total']
        warp_correct_count = overall_results['warped']['correct']
        
        overall_warp_acc = warp_correct_count / max(1, warp_total)
        # Removed baseline delta logging
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"SAMPLE: {sample_id} - {timestamp}\n")
            f.write("-"*80 + "\n")
            f.write(f"QUESTION: {question}\n\n")
            f.write(f"GROUND TRUTH ANSWERS: {ground_truth_answers}\n\n")
            f.write(f"WARPED IMAGE ANSWER:   {warped_answer} (Correct: {warped_correct})\n\n")
            f.write("CURRENT OVERALL STATS:\n")
            f.write(f"  Warped accuracy:   {overall_warp_acc:.4f} ({warp_correct_count}/{warp_total})\n")
            # Removed baseline delta logging
    except Exception as e:
        print(f"Error updating detailed log for sample {sample_data.get('sample_id', 'UNKNOWN')}: {e}")
# --- End Logging Functions ---

# --- Accuracy Calculation Helper ---
def calculate_vqa_accuracy(predicted_answer, ground_truth_answers, threshold=3):
    """
    Calculates VQA accuracy based on standard metric.
    Args:
        predicted_answer (str): The model's predicted answer.
        ground_truth_answers (list): List of ground truth answer strings.
        debug (bool): Whether to print debug information.
        log_file (str): Path to debug log file.
    Returns:
        float: VQA accuracy score between 0.0 and 1.0
    """
    if not predicted_answer or not ground_truth_answers:
        return False
    
    # Process the predicted answer
    processed_pred = process_text(predicted_answer)
    
    # Process ground truth answers
    processed_gts = [process_text(gt) for gt in ground_truth_answers]
        
    # count = ground_truth_answers.count(predicted_answer)
    match_count = sum(1 for gt in processed_gts if gt == processed_pred)
    
    # VQA accuracy: min(count / 3, 1) >= 1 -> count >= 3
    return match_count >= threshold

# --- Main Evaluation Function --- 
def evaluate_textvqa_accuracy(model_path="liuhaotian/llava-v1.5-7b"):
    """Evaluate the accuracy of LLaVA on TextVQA for original and warped images"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    moving_avg_log = os.path.join(OUTPUT_DIR, f"textvqa_moving_accuracy_{timestamp}.csv")
    detailed_log = os.path.join(OUTPUT_DIR, f"textvqa_detailed_log_{timestamp}.txt")
    debug_log = os.path.join(OUTPUT_DIR, f"textvqa_debug_preprocessing_{timestamp}.txt")
    
    # Initialize detailed log file
    try:
        with open(detailed_log, 'w', encoding='utf-8') as f:
            f.write(f"TEXTVQA FULL EVALUATION - DETAILED LOG\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Evaluation Metric: Standard VQA accuracy with text preprocessing (min(0.3*n, 1), n=matching answers)\n")
            f.write("="*80 + "\n\n")
    except Exception as e:
        print(f"Error initializing detailed log file {detailed_log}: {e}")
        sys.exit(1)
        
    # Load model
    tokenizer, model, image_processor, context_len, model_name = load_llava_model(model_path)
    
    # Prepare result containers (Overall only)
    overall_results = {
        'warped': {'correct': 0, 'total': 0}
    }
    detailed_results_list = [] # Store per-sample results for final JSON
    sample_counter = 0
    
    # Collect all metadata files (no categories)
    all_metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    
    if not all_metadata_files:
        print(f"Error: No metadata JSON files found in {METADATA_DIR}")
        sys.exit(1)

    print(f"\nFound {len(all_metadata_files)} metadata files in {METADATA_DIR}")
    print(f"Starting evaluation using standard VQA accuracy (count >= 3)...")
    print(f"Detailed log will be written to: {detailed_log}")
    print(f"Moving average log will be written to: {moving_avg_log}")
    
    for metadata_file in tqdm(all_metadata_files, desc="Evaluating TextVQA samples"):
        try:
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extract information
            sample_id = metadata.get('sample_id')
            question = metadata.get('question')
            ground_truth_answers = metadata.get('answers') # List of answers
            saved_paths = metadata.get('saved_paths', {}) # Get saved paths dict
            warped_image_path = saved_paths.get('warped_image_identity')  # Use the warped image path
            
            # Basic validation
            if not sample_id:
                print(f"Warning: Skipping metadata file {metadata_file} due to missing 'sample_id'")
                continue
            if not question:
                print(f"Warning: Skipping sample {sample_id} due to missing 'question'")
                continue
            if not ground_truth_answers or not isinstance(ground_truth_answers, list):
                print(f"Warning: Skipping sample {sample_id} due to missing or invalid 'answers' list")
                continue
            # Ensure GT answers are strings for comparison
            ground_truth_answers = [str(gt) for gt in ground_truth_answers]
            if not warped_image_path or not os.path.exists(warped_image_path):
                print(f"Warning: Skipping sample {sample_id} due to missing/invalid warped image path: {warped_image_path}")
                continue
                
            # --- Evaluate Warped Image --- 
            warped_answer_raw = get_llava_response(model, tokenizer, image_processor, 
                                               warped_image_path, question)
            
            # Skip if error during inference
            if warped_answer_raw is None:
                print(f"Warning: Skipping sample {sample_id} due to inference error.")
                continue
            
            # --- Process Answers and Check Correctness --- 
            warped_answer = warped_answer_raw.strip()
            ground_truth_answers = [gt.strip() for gt in ground_truth_answers]
            
            # Calculate VQA Accuracy (now with threshold=1)
            warped_correct = calculate_vqa_accuracy(warped_answer, ground_truth_answers, threshold=1)
            
            # --- Update Overall Results --- 
            overall_results['warped']['total'] += 1
            overall_results['warped']['correct'] += warped_correct
            
            # --- Prepare Detailed Log Entry --- 
            sample_data = {
                'sample_id': sample_id,
                'question': question,
                'ground_truth_answers': ground_truth_answers,  # Keep original case for log
                'warped_answer': warped_answer_raw,            # Log raw answer
                'warped_correct': warped_correct
            }
            detailed_results_list.append(sample_data)
            
            # --- Update Counter ---
            sample_counter += 1
            update_detailed_log(detailed_log, sample_data, overall_results)
            # Update moving average log periodically (e.g., every 10 samples)
            if sample_counter % 10 == 0:
                update_moving_accuracy_log(moving_avg_log, overall_results, sample_counter)
            
            # Save interim results periodically (e.g., every 100 samples)
            if sample_counter % 100 == 0:
                interim_warp_acc = overall_results['warped']['correct'] / max(1, overall_results['warped']['total'])
                interim_results_data = {
                    'model': model_path,
                    'overall_warped_accuracy': interim_warp_acc,
                    'total_samples_processed': overall_results['warped']['total'],
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                interim_path = os.path.join(OUTPUT_DIR, f"textvqa_interim_results_{timestamp}_{sample_counter}.json")
                try:
                    with open(interim_path, 'w') as f_interim:
                        json.dump(interim_results_data, f_interim, indent=2)
                except Exception as e_interim:
                    print(f"Warning: Could not save interim results at sample {sample_counter}: {e_interim}")
            # --- End Log Updates --- 
            
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {metadata_file}. Skipping.")
            continue
        except FileNotFoundError:
             print(f"Error: Metadata file not found: {metadata_file}. Skipping. (Should not happen if glob worked)")
             continue
        except Exception as e:
            print(f"Unexpected error processing metadata file {metadata_file}: {e}. Skipping.")
            # Consider adding a counter for these unexpected skips
            continue
    # --- End Main Loop --- 

    # (Moving-average log removed along with original accuracy metrics)
    
    final_warp_total = overall_results['warped']['total']
    final_warp_correct = overall_results['warped']['correct']

    if final_warp_total == 0:
        print("\nError: No samples were successfully processed.")
        overall_warp_acc = 0.0
    else:
        overall_warp_acc = final_warp_correct / final_warp_total

    # --- Prepare Final Results ---
    final_results = {
        'model': model_path,
        'dataset': 'TextVQA',
        'evaluation_metric': 'Standard VQA accuracy with text preprocessing (min(0.3*n, 1), n=matching answers)',
        'overall_warped_accuracy': overall_warp_acc,
        'total_samples_evaluated': final_warp_total,
        'detailed_results': detailed_results_list
    }
    
    # --- Save Final Results --- 
    results_path = os.path.join(OUTPUT_DIR, f"textvqa_accuracy_{timestamp}.json")
    summary_path = os.path.join(OUTPUT_DIR, f"textvqa_accuracy_{timestamp}_summary.txt")
    
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving final JSON results to {results_path}: {e}")
        
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"TextVQA Accuracy Evaluation with LLaVA\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Total samples evaluated: {final_warp_total}\n")
            f.write(f"Evaluation Metric: {final_results['evaluation_metric']}\n\n")
            f.write(f"Overall accuracy on warped images: {overall_warp_acc:.4f} ({final_warp_correct}/{final_warp_total})\n")
    except Exception as e:
        print(f"Error saving summary file to {summary_path}: {e}")

    # Add final summary to detailed log
    try:
        with open(detailed_log, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("EVALUATION COMPLETE - FINAL SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total samples evaluated: {final_warp_total}\n")
            f.write(f"Evaluation Metric: {final_results['evaluation_metric']}\n\n")
            f.write(f"Overall accuracy on warped images: {overall_warp_acc:.4f} ({final_warp_correct}/{final_warp_total})\n")
    except Exception as e:
        print(f"Error appending final summary to detailed log {detailed_log}: {e}")

    print(f"\nEvaluation complete!")
    if final_warp_total > 0:
        print(f"  Overall Warped Accuracy: {overall_warp_acc:.4f}")
    print(f"Results JSON saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Moving average log saved to: {moving_avg_log}")
    print(f"Detailed log saved to: {detailed_log}")
    
    return summary_path, moving_avg_log, detailed_log

# --- Entry Point --- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA accuracy on TextVQA dataset (original vs warped)")
    parser.add_argument("--model", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="LLaVA model path to use for evaluation (e.g., liuhaotian/llava-v1.5-7b)")
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_textvqa_accuracy(args.model) 
