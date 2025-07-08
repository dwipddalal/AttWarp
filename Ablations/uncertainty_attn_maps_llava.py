"""
Utility wrappers for LLaVA attention maps (absolute + relative, arbitrary layer).

Import this module **after** the original `llava_methods.py` is on PYTHONPATH.
"""

from typing import Dict, List
import cv2, torch, numpy as np
from PIL import Image
from llava_methods import IMAGE_TOKEN_INDEX, NUM_PATCHES   # constants

# ---------------------------------------------------------------------------
# Because rel_attention_llava() is hard-wired to ATT_LAYER=14, we re-implement
# the low-level attention extraction so we can choose any layer L.
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_abs_rel_maps_llava(img_path: str,
                           question: str,
                           layers: List[int],
                           model,
                           processor) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Return dicts identical to the Qwen helper:
        {"absolute":{L:full}, "relative":{L:full},
         "absolute_small":{L:18x27}, "relative_small":{L:18x27}}
    where `full` is H×W (original resolution).
    """
    # build prompts ---------------------------------------------------------
    prompt = "<image>\nUSER: " + question + \
             " Answer the question using a single word or phrase.\nASSISTANT:"
    gen_prompt = ("<image>\nUSER: Write a general description of the image. "
                  "Answer the question using a single word or phrase.\nASSISTANT:")

    img_pil = Image.open(img_path).convert("RGB")
    W_img, H_img = img_pil.size

    # ----------------------------------------------------------------------
    # preprocess once for user prompt & general prompt
    inp = processor(prompt, img_pil, return_tensors="pt", padding=True
                    ).to(model.device, torch.bfloat16)
    gen = processor(gen_prompt, img_pil, return_tensors="pt", padding=True
                    ).to(model.device, torch.bfloat16)

    # forward once ---------------------------------------------------------
    out = model(**inp, output_attentions=True)
    gnt = model(**gen, output_attentions=True)

    # locate vision span
    seq = inp["input_ids"][0].tolist()
    pos = seq.index(IMAGE_TOKEN_INDEX)               # first image token index
    img_span = slice(pos, pos + NUM_PATCHES**2)

    maps = {"absolute":{}, "relative":{},
            "absolute_small":{}, "relative_small":{}}

    for L in layers:
        idx = L - 1
        att_abs  = out.attentions[idx][0, :, -1, img_span].mean(0)
        att_gen  = gnt.attentions[idx][0, :, -1, img_span].mean(0)
        abs_map  = att_abs.float().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
        rel_map  = abs_map / (att_gen.float().cpu().numpy()
                              .reshape(NUM_PATCHES, NUM_PATCHES) + 1e-8)

        for typ, grid in (("absolute", abs_map), ("relative", rel_map)):
            full  = cv2.resize(grid, (W_img, H_img), interpolation=cv2.INTER_LANCZOS4)
            small = cv2.resize(grid, (18, 27), interpolation=cv2.INTER_LANCZOS4)
            maps[typ][L] = full
            maps[f"{typ}_small"][L] = small

    del out, gnt, inp, gen
    torch.cuda.empty_cache()
    return maps
#!/usr/bin/env python3
# --------------------------------------------------------------
#  eval_attention_layers_llava.py
# --------------------------------------------------------------
"""
Run the TextVQA attention-alignment experiment with **LLaVA-1.5-7B**.

Outputs:
  /home/skan/appendix_attention_maps/
        results_llava.csv
        summary_llava.txt
        per_image/<dataset_id>/            (same folders as Qwen)
            ├─ <id>_question.txt           (only written if absent)
            ├─ <id>_orig.png               (copied once)
            ├─ <id>_<layer>_absolute.png   (18×27 heat-map)
            └─ <id>_<layer>_relative.png
"""

import os, json, glob, csv, random
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from tqdm import tqdm

# ----------------------------- paths & constants ---------------------------
IMG_DIR   = "/home/skan/text_vqa/outputs_all/exp21_rt_def_all/original_images"
ANN_PATH  = "/home/skan/text_vqa_bounding_boxes_madhav/textvqa_annotations.json"
TEXTVQA_JSON = "/home/skan/text_vqa/TextVQA_0.5.1_val.json"

LAYERS   = [4, 8, 10, 12, 14, 16, 18, 20, 24]
SAMPLE_N = 1000
SEED     = 42

OUT_ROOT = Path("/home/skan/appendix_attention_maps")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_ROOT / "results_llava.csv"
TXT_PATH = OUT_ROOT / "summary_llava.txt"
IMG_OUT_ROOT = OUT_ROOT / "per_image"
IMG_OUT_ROOT.mkdir(exist_ok=True)

# ----------------------------- load model ----------------------------------
from transformers import AutoProcessor, LlavaForConditionalGeneration
print("Loading LLaVA-1.5-7B … (may take ~1-2 min)")
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, attn_implementation="eager"
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ----------------------------- helper imports ------------------------------

# bbox helpers --------------------------------------------------------------
BBOX_METHODS = ["xywh", "xyxy"]
def interpret_bbox(b, m):
    if m=="xywh":
        x,y,w,h=b; return int(x),int(y),int(w),int(h)
    x1,y1,x2,y2=b; return int(x1),int(y1),int(x2-x1),int(y2-y1)
def top1(att, box):
    x0,y0,w,h=box; r,c=np.unravel_index(np.argmax(att),att.shape)
    return 1 if (x0<=c<x0+w and y0<=r<y0+h) else 0
def am_all(att, box):
    x0,y0,w,h=box; return att[y0:y0+h, x0:x0+w].sum()/(att.sum()+1e-12)

# ----------------------------- question lookup -----------------------------
def build_q_lk(p):
    data=json.load(open(p))
    return {str(d["question_id"]):d["question"] for d in data["data"]}
QUESTION_LK = build_q_lk(TEXTVQA_JSON)

# ----------------------------- load bbox & sample --------------------------
ann = json.load(open(ANN_PATH))
bbox_lookup={str(x["dataset_id"]):x["bbox"] for x in ann}
ids=list(bbox_lookup.keys()); random.Random(SEED).shuffle(ids)
sample_ids=ids[:SAMPLE_N] if len(ids)>=SAMPLE_N else ids
print(f"Sample size: {len(sample_ids)}  images")

# ----------------------------- metric buckets ------------------------------
metrics={t:{L:{"top1":[],"am":[]} for L in LAYERS}
         for t in ("absolute","relative")}

# ----------------------------- main loop -----------------------------------
with CSV_PATH.open("w",newline="") as csvf:
    wr=csv.writer(csvf); wr.writerow(["dataset_id","layer","type","top1","am"])
    for did in tqdm(sample_ids,desc="LLaVA eval"):
        pngs=glob.glob(f"{IMG_DIR}/{did}_*.png")
        if not pngs: continue
        img_path=pngs[0]

        question=QUESTION_LK.get(did,"What is shown?")
        print(f"Question: {question}")
        try:
            maps=get_abs_rel_maps_llava(img_path,question,LAYERS,model,processor)
        except Exception as e:
            print(f"  ⚠️  {did}  skipped ({e})"); continue

        # bbox
        b_raw=bbox_lookup[did]; box=None
        for m in BBOX_METHODS:
            b=interpret_bbox(b_raw,m)
            if b[2]>0 and b[3]>0: box=b; break
        if box is None: continue

        # per-image folder
        pdir=IMG_OUT_ROOT/did; pdir.mkdir(parents=True,exist_ok=True)
        (pdir/f"{did}_question.txt").write_text(question+"\n",encoding="utf-8") \
            if not (pdir/f"{did}_question.txt").exists() else None
        if not (pdir/f"{did}_orig.png").exists():
            cv2.imwrite(str(pdir/f"{did}_orig.png"),cv2.imread(img_path))

        # per layer/type metrics + heat-maps
        for typ in ("absolute","relative"):
            for L in LAYERS:
                full=maps[typ][L]; small=maps[f"{typ}_small"][L]
                t=top1(full,box); a=am_all(full,box)
                metrics[typ][L]["top1"].append(t)
                metrics[typ][L]["am"].append(a)
                wr.writerow([did,L,typ,t,a])

                fname=pdir/f"{did}_{L}_{typ}.png"
                if not fname.exists():
                    amin,amax=small.min(),small.max()
                    norm=(small-amin)/(amax-amin+1e-9)
                    heat=cv2.applyColorMap((norm*255).astype(np.uint8),
                                           cv2.COLORMAP_VIRIDIS)
                    cv2.imwrite(str(fname),heat)

# ----------------------------- summary -------------------------------------
with TXT_PATH.open("w") as f:
    hdr=f"{'Layer':>6}  {'Type':>9}  {'Top-1':>8}  {'AM@all':>8}\n"
    f.write(hdr); f.write("-"*len(hdr)+"\n")
    print("\n"+hdr.strip()); print("-"*len(hdr))
    for L in LAYERS:
        for typ in ("absolute","relative"):
            tops=metrics[typ][L]["top1"]; ams=metrics[typ][L]["am"]
            line=f"{L:>6}  {typ:>9}  {np.mean(tops):8.3f}  {np.mean(ams):8.3f}\n"
            f.write(line); print(line.strip())

print(f"\nLLaVA CSV   → {CSV_PATH}")
print(f"LLaVA table → {TXT_PATH}")
print("Done.")
