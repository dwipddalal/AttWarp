"""
MarginalNet training on the "FullDataset" (GQA/TextVQA/DocVQA) setup.

Refactored from `gt_warp_prediction_full_Dataset.py` into smaller, testable modules.
"""

from .config import MarginalNetFullDatasetConfig
from .trainer import train_loop

__all__ = ["MarginalNetFullDatasetConfig", "train_loop"]


