"""
Attention extraction module for LLaVA models.

This module provides utilities for extracting attention maps from LLaVA models
without relying on external libraries like apiprompting.
"""

from .functions import getmask, getmask_batch, get_model
from .llava import (
    hook_logger,
    MaskHookLogger,
    batch_hook_logger,
    BatchMaskHookLogger,
    blend_mask,
    llava_api,
)

__all__ = [
    'getmask',
    'getmask_batch',
    'get_model',
    'hook_logger',
    'MaskHookLogger',
    'batch_hook_logger',
    'BatchMaskHookLogger',
    'blend_mask',
    'llava_api',
]
