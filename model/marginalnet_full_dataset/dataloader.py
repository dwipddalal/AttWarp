from __future__ import annotations

import json
import os
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

import pdb

GQA_CATEGORY_TO_TRANSFORM: Dict[str, str] = {
    "query_rel": "sqrt",
    "query_attr": "sqrt",
    "verify_rel": "sqrt",
    "logical_attr": "none",
    "query_cat": "sqrt",
    "verify_attr": "none",
    "choose_attr": "iden",
    "logical_obj": "iden",
    "compare_attr": "none",
    "choose_rel": "sqrt",
    "verify_obj": "sqrt",
    "query_global": "sqrt",
    "choose_cat": "iden",
    "verify_global": "none",
    "choose_global": "sqrt",
}

class FullDataset(Dataset):
    """PyTorch Dataset that mixes GQA, TextVQA, and DocVQA samples from logs-based folders.

    Folder structure per dataset root:
        - logs/       (JSON logs; one per question-image pair)
        - npy/
        - overlays/
        - raw/

    Returns a dictionary with keys (tensors may be variable-sized if ``image_size`` is None):
        - image (FloatTensor): (3, H, W) in [0, 1]
        - attention_map (FloatTensor): (1, H, W) in [0, 1]
        - question (str)
        - answer (str)
        - dataset (str): "gqa" or "textvqa" or "docvqa"
        - question_id (str|None)
        - saved_warped_image (FloatTensor):
            - for DocVQA: identity (same as image)
            - for others: zero tensor matching image shape (compat)
    """

    def __init__(
        self,
        gqa_root_dir: str = "Dataset_for_training/gqa_qwen_multilayer",
        textvqa_root_dir: str = "Dataset_for_training/textvqa_qwen_multilayer",
        docvqa_root_dir: str = "Dataset_for_training/docvqa_qwen_multilayer",
        num_samples_per_dataset: int = 15000,
        artifact_type: str = "relative",
        artifact_layer: int = 16,
        random_seed: int = 42,
        image_size: Optional[int] = None,
        ):  # noqa: D401
        super().__init__()

        if artifact_type not in {"relative", "absolute"}:
            raise ValueError("artifact_type must be 'relative' or 'absolute'")
        self.artifact_type = artifact_type
        self.artifact_layer = int(artifact_layer)

        rng = random.Random(random_seed)

        # Collect entries from logs/metadata.jsonl under each dataset root
        def _collect_entries(root_dir: str) -> List[Dict[str, Any]]:
            logs_dir = os.path.join(root_dir, "logs")
            if not os.path.isdir(logs_dir):
                print(f"No logs directory found in {root_dir}")
                return []
            jsonl_path = os.path.join(logs_dir, "metadata.jsonl")
            entries: List[Dict[str, Any]] = []

            with open(jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            entries.append(obj)
                    except Exception:
                        continue
            return entries

        gqa_entries_all: List[Dict[str, Any]] = _collect_entries(gqa_root_dir)
        textvqa_entries_all: List[Dict[str, Any]] = _collect_entries(textvqa_root_dir)
        docvqa_entries_all: List[Dict[str, Any]] = _collect_entries(docvqa_root_dir)

        if (
            len(gqa_entries_all) < num_samples_per_dataset
            or len(textvqa_entries_all) < num_samples_per_dataset
            or len(docvqa_entries_all) < num_samples_per_dataset
        ):
            print(f"GQA logs: {len(gqa_entries_all)}")
            print(f"TextVQA logs: {len(textvqa_entries_all)}")
            print(f"DocVQA logs: {len(docvqa_entries_all)}")
            raise ValueError("Not enough log samples in one of the datasets to satisfy requested num_samples_per_dataset")

        self.gqa_entries = rng.sample(gqa_entries_all, num_samples_per_dataset)
        self.textvqa_entries = rng.sample(textvqa_entries_all, num_samples_per_dataset)
        self.docvqa_entries = rng.sample(docvqa_entries_all, num_samples_per_dataset)

        # Build unified index list and shuffle
        self.samples: List[Tuple[Dict[str, Any], str]] = [
            *( (entry, "gqa") for entry in self.gqa_entries ),
            *( (entry, "textvqa") for entry in self.textvqa_entries ),
            *( (entry, "docvqa") for entry in self.docvqa_entries ),
        ]
        rng.shuffle(self.samples)

        # Transforms: if image_size is provided, resize to a fixed square; else keep native size
        self.image_size = int(image_size) if image_size is not None else None
        if self.image_size is not None:
            self.img_transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
            ])
            self.attn_transform = T.Compose([
                T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),  # converts to (1, H, W) in [0,1]
            ])
        else:
            self.img_transform = T.ToTensor()
            # For attention when variable-sized: we will resize to the image size on-the-fly in __getitem__
            self.attn_transform = None

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _select_artifact(artifacts: List[Dict[str, Any]], desired_layer: int, desired_type: str) -> Optional[Dict[str, Any]]:
        if not artifacts:
            return None
        same_type = [a for a in artifacts if str(a.get("type", "")).lower() == desired_type]
        if not same_type:
            same_type = artifacts  # fallback to any type
        # try exact layer
        for a in same_type:
            if int(a.get("layer", -1)) == int(desired_layer):
                return a
        # else pick the highest available layer among same_type
        try:
            return sorted(same_type, key=lambda x: int(x.get("layer", -1)))[-1]
        except Exception:
            return same_type[0]

    @staticmethod
    def _load_attention_map_from_artifact(artifact: Dict[str, Any]) -> np.ndarray:
        # Prefer processed npy_path, then raw_path, then overlay image
        npy_path = FullDataset._resolve_file_path(artifact.get("npy_path"))
        raw_path = FullDataset._resolve_file_path(artifact.get("raw_path"))
        overlay_path = FullDataset._resolve_file_path(artifact.get("overlay_path"))

        if npy_path and os.path.isfile(npy_path):
            arr = np.load(npy_path)
        elif raw_path and os.path.isfile(raw_path):
            arr = np.load(raw_path)
        elif overlay_path and os.path.isfile(overlay_path):
            with Image.open(overlay_path) as _im:
                im_gray = _im.convert("L")
                arr = np.asarray(im_gray.copy(), dtype=np.float32) / 255.0
                return arr
        else:
            raise FileNotFoundError("No valid artifact path found (npy/raw/overlay)")

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            # convert to single-channel by averaging if needed
            arr = arr.mean(axis=2)
        # robust normalization to [0,1]
        max_val = float(arr.max()) if arr.size else 0.0
        if max_val > 0:
            if max_val > 1.0:
                arr = arr / max_val
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    @staticmethod
    def _resolve_file_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isfile(path):
            return path
        # Try replacing '/CLIP/' with '/data/' once at the root level
        if "CLIP/" in path:
            alt = path.replace("CLIP/", "data/", 1)
            if os.path.isfile(alt):
                return alt
        return None

    def __getitem__(self, idx: int):
        meta, dataset_name = self.samples[idx]

        question: str = meta.get("question", "")
        answer_val = meta.get("answer", None)
        answer = answer_val if isinstance(answer_val, str) and answer_val is not None else ""

        # Load image
        img_path_raw = meta.get("image_path")
        img_path = FullDataset._resolve_file_path(img_path_raw)
        if not img_path:
            raise FileNotFoundError(f"Image path not found: {img_path_raw}")
        with Image.open(img_path) as _im:
            im_rgb = _im.convert("RGB")
            image = im_rgb.copy()

        # Select artifact and load attention map
        artifacts = meta.get("artifacts", [])
        art = self._select_artifact(artifacts, self.artifact_layer, self.artifact_type)
        if art is None:
            raise FileNotFoundError("No artifacts listed in log JSON")
        attn_np = self._load_attention_map_from_artifact(art)

        # To tensors (resize only if a fixed size is configured)
        if self.image_size is not None:
            image_tensor = self.img_transform(image)
            attn_img = Image.fromarray((attn_np * 255).astype(np.uint8))
            attn_tensor = self.attn_transform(attn_img)  # (1, image_size, image_size)
        else:
            # Keep native resolution; ensure attention matches image size
            image_tensor = self.img_transform(image)
            attn_img = Image.fromarray((attn_np * 255).astype(np.uint8))
            if attn_img.size != image.size:
                attn_img = attn_img.resize(image.size, resample=Image.NEAREST)
            attn_tensor = T.ToTensor()(attn_img)

        # saved_warped_image: identity for DocVQA, zero for others
        if dataset_name == "docvqa":
            warped_tensor = image_tensor.clone()
        else:
            warped_tensor = torch.zeros_like(image_tensor)

        # best-effort question id extraction
        qid = (
            meta.get("question_id", None)
            or meta.get("questionId", None)
            or meta.get("qid", None)
            or meta.get("id", None)
        )

        # dataset field: prefer from meta if present
        dset = str(meta.get("dataset", dataset_name))

        return {
            "image": image_tensor,
            "attention_map": attn_tensor,
            "saved_warped_image": warped_tensor,
            "question": question,
            "answer": answer,
            "dataset": dset,
            "question_id": qid,
            # Best-effort: expose GQA structural×semantic bucket/category if present in logs
            # Downstream will decide how to use it (e.g., transform overrides)
            "bucket": (
                meta.get("bucket", None)
                or meta.get("category", None)
                or meta.get("gqa_bucket", None)
                or meta.get("question_type", None)
            ) if (isinstance(dset, str) and ("gqa" in dset.lower())) else None,
        }

class MixedGQATextVQADataset(Dataset):
    """PyTorch Dataset that mixes GQA and TextVQA samples.

    It returns a dictionary with the following keys:
        - image (FloatTensor): image tensor in range [0, 1] of shape (3, H, W)
        - attention_map (FloatTensor): attention map tensor (1, H, W) rescaled to image size
        - question (str)
        - answer (str)
        - dataset (str): either "gqa" or "textvqa"
    """

    def __init__(
        self,
        gqa_metadata_dir: str = "CLIP/results/gqa_processed/metadata",
        textvqa_metadata_dir: str = "CLIP/results/textvqa_processed_full/metadata",
        num_samples_per_dataset: int = 2000,
        image_size: int = 224,
        random_seed: int = 42,
        ):  # noqa: D401
        super().__init__()

        self.image_size = image_size
        rng = random.Random(random_seed)

        # Collect metadata paths
        gqa_meta_all: List[str] = [
            os.path.join(gqa_metadata_dir, f)
            for f in os.listdir(gqa_metadata_dir)
            if f.endswith("_metadata.json")
        ]
        textvqa_meta_all: List[str] = [
            os.path.join(textvqa_metadata_dir, f)
            for f in os.listdir(textvqa_metadata_dir)
            if f.endswith("_metadata.json")
        ]

        if len(gqa_meta_all) < num_samples_per_dataset or len(textvqa_meta_all) < num_samples_per_dataset:
            raise ValueError("Not enough samples in one of the datasets to satisfy requested num_samples_per_dataset")

        self.gqa_meta_paths = rng.sample(gqa_meta_all, num_samples_per_dataset)
        self.textvqa_meta_paths = rng.sample(textvqa_meta_all, num_samples_per_dataset)

        # Build unified index list and shuffle
        self.samples: List[Tuple[str, str]] = [
            *( (path, "gqa") for path in self.gqa_meta_paths ),
            *( (path, "textvqa") for path in self.textvqa_meta_paths ),
        ]
        rng.shuffle(self.samples)

        # Transforms
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        self.attn_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),  # converts to (1, H, W)
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_attention_map(self, saved_paths: Dict[str, Any]) -> np.ndarray:
        """Attempt to load an attention map as a numpy array."""
        npy_path: Optional[str] = saved_paths.get("mota_mask_npy") or saved_paths.get("raw_attention_map_npy")
        if npy_path and os.path.isfile(npy_path):
            return np.load(npy_path)

        img_path = (
            saved_paths.get("mota_mask_visualization")
            or saved_paths.get("attention_map_image_from_api")
        )
        if img_path and os.path.isfile(img_path):
            img = Image.open(img_path).convert("L")  # grayscale
            return np.asarray(img) / 255.0

        raise FileNotFoundError("No attention map found for sample")

    def __getitem__(self, idx: int):
        meta_path, dataset_name = self.samples[idx]
        with open(meta_path, "r") as f:
            meta = json.load(f)

        question: str = meta["question"]
        if dataset_name == "gqa":
            answer = meta["answer"]
        else:  # textvqa
            answers = meta.get("answers", [])
            answer = answers[0] if answers else ""

        saved_paths = meta["saved_paths"]
        # Load image
        img_path = saved_paths["original_image"]
        image = Image.open(img_path).convert("RGB")

        attn_np = self._load_attention_map(saved_paths)
        attn_img = Image.fromarray((attn_np * 255).astype(np.uint8)) if isinstance(attn_np, np.ndarray) else attn_np

        image_tensor = self.img_transform(image)
        attn_tensor = self.attn_transform(attn_img)

        warped_path = (
            saved_paths.get("warped_image_identity")
            or saved_paths.get("warped_image")
            or saved_paths.get("saved_warped_image")
        )
        if warped_path and os.path.isfile(warped_path):
            warped_img = Image.open(warped_path).convert("RGB")
            warped_tensor = self.img_transform(warped_img)
        else:
            warped_tensor = torch.zeros_like(image_tensor)

        qid = (
            meta.get("question_id", None)
            or meta.get("questionId", None)
            or meta.get("qid", None)
            or meta.get("id", None)
        )

        return {
            "image": image_tensor,
            "attention_map": attn_tensor,
            "saved_warped_image": warped_tensor,
            "question": question,
            "answer": answer,
            "dataset": dataset_name,
            "question_id": qid,
        }


if __name__ == "__main__":
    ds = MixedGQATextVQADataset()
    for i in range(5):
        sample = ds[i]
        print(
            {
                "idx": i,
                "dataset": sample["dataset"],
                "question": sample["question"],
                "answer": sample["answer"],
                "image_shape": sample["image"].shape,
                "attention_map_shape": sample["attention_map"].shape,
            }
        )
