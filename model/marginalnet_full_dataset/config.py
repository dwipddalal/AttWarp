from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MarginalNetFullDatasetConfig:
    # ---- W&B ----
    wandb_project: str = os.getenv("WANDB_PROJECT", "attwarp-marginalnet")
    wandb_entity: str | None = os.getenv("WANDB_ENTITY")  # optional team/org
    log_every: int = int(os.getenv("LOG_EVERY", "20"))

    # ---- data ----
    gqa_root_dir: str = "/shared/nas2/dwip2/data/Dataset_for_training/gqa_qwen_multilayer"
    textvqa_root_dir: str = "/shared/nas2/dwip2/data/Dataset_for_training/textvqa_qwen_multilayer"
    docvqa_root_dir: str = "/shared/nas2/dwip2/data/Dataset_for_training/docvqa_qwen_multilayer"
    num_per_ds: int = 12000
    image_size: int = 512

    # ---- reproducibility ----
    seed: int = 13
    eps: float = 1e-6

    # ---- model ----
    hidden: int = 256

    # ---- training ----
    epochs: int = 50
    batch_size: int = 128
    lr: float = 3e-4
    wd: float = 1e-4
    grad_clip: float = 1.0
    workers: int = 4
    cpu: bool = False

    # ---- losses ----
    w_cdf: float = 10.0
    axis_len: int = 256

    # ---- warmup / stabilizers ----
    warmup_steps: int = 1000
    alpha0: float = 0.0
    alpha_decay_steps: int = 2000
    ent_weight: float = 1e-3

    # ---- LLaVA ----
    llava_repo_root: str = "/shared/nas2/dwip2/training_Attwarp"
    llava_model: str = "liuhaotian/llava-v1.5-7b"

    # ---- experiments ----
    experiments_root: str = "/shared/nas2/dwip2/training_Attwarp/Experiments"
    vis_every: int = 200


