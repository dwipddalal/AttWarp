from __future__ import annotations

from typing import Any, Dict, List

import torch

from dataloader import GQA_CATEGORY_TO_TRANSFORM


def collate_str(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B,3,H,W)
    atts = torch.stack([b["attention_map"] for b in batch], dim=0)  # (B,1,H,W)
    q_str = [b["question"] for b in batch]
    ans = [b["answer"] for b in batch]
    dset = [b["dataset"] for b in batch]
    buckets = [b.get("bucket", None) for b in batch]
    return {
        "image": images,
        "attention_map": atts,
        "q_str": q_str,
        "answer": ans,
        "dataset": dset,
        "bucket": buckets,
    }


def load_gqa_transform_map(_: str = "") -> Dict[str, str]:
    # Sourced from dataloader.GQA_CATEGORY_TO_TRANSFORM
    return dict(GQA_CATEGORY_TO_TRANSFORM)


def transform_for_sample(dataset_name: Any, bucket: Any, mapping: Dict[str, str]) -> str:
    """
    Returns transform name for a single sample:
    - GQA: based on bucket->transform map (fallback 'sqrt')
    - others: 'iden'
    """
    try:
        ds = str(dataset_name).lower() if dataset_name is not None else ""
        if "gqa" in ds:
            b = str(bucket) if bucket is not None else None
            if b in mapping:
                return mapping[b]
            return "sqrt"
        return "iden"
    except Exception:
        return "iden"


