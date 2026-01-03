# Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/6874350)
[![Project](https://img.shields.io/badge/Project-Page-black)](https://dwipddalal.github.io/Attwarp/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18GHH-SXyvfvrDUiZiuqaBuF7Ws2nierM#scrollTo=VWZ9cGqnkkW9)

## Overview

This repository contains the code, model weights, and environment setup for the paper **Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping**. The project focuses on attention-guided image warping to improve multimodal large language model (MLLM) performance.

<img width="1683" height="609" alt="Overview" src="https://github.com/user-attachments/assets/7ea03106-4615-4271-b931-e803be272f28" />

## Contents

- [MLLM Models](#mllm-models)
- [Attention Map Extraction](#attention-map-extraction)
- [Setup](#setup)
- [Quick Setup Check](#quick-setup-check)
- [Qualitative Results](#qualitative-results)
- [Colab Demo](#colab-demo)
- [Contact](#contact)
- [BibTeX](#bibtex)
- [License](#license)

## MLLM Models

The code and environment setup for the MLLM models used in this work are hosted by their respective projects:

- **LLaVA**: [Official code](https://github.com/haotian-liu/LLaVA)
- **Qwen**: [Official code](https://github.com/QwenLM/Qwen3-VL)

## Attention Map Extraction

We will soon release our attention map extraction utilities. In the meantime, you can use the implementation from:
https://github.com/saccharomycetes/mllms_know

## Setup

From the repository root, create the conda environment once and then activate it whenever you work on the model code:

```bash
conda env create -f attwarp.yaml
conda activate attwarp
```

If you already created the environment, just run the activation step.

## Quick Setup Check

```bash
cd /path/to/AttWarp
python "Attention Guided Warping"/main.py
python "Attention Guided Warping"/TextVQA/evaluate_accuracy.py
```

## Model
This directory contains the training code for the MarginalNet full-dataset setup. Use the tree below as a map of how the modules connect.

```
model/
├── __init__.py
└── marginalnet_full_dataset/
    ├── __init__.py                -> Exports MarginalNetFullDatasetConfig + train_loop.
    ├── cli.py                     -> CLI entrypoint; builds config and calls trainer.train_loop.
    ├── config.py                  -> Dataclass for configuration defaults and env overrides.
    ├── trainer.py                 -> Orchestrates training loop; wires data, model, losses, logging.
    ├── data.py                    -> Dataset transforms/collation helpers used by trainer.
    ├── dataloader.py              -> Dataset definitions and sampling utilities (FullDataset).
    ├── model.py                   -> MarginalNet architecture + core tensor ops.
    ├── losses.py                  -> Loss functions used during training.
    ├── checkpoint_utils.py        -> Warp/CDF helpers, plotting utilities for checkpoints.
    ├── plots.py                   -> Training/validation curve plots.
    ├── experiment.py              -> Run directory creation and experiment bookkeeping.
    └── wandb_utils.py             -> Weights & Biases initialization and logging helpers.
```

### Training hyperparameters

The MarginalNet full-dataset training setup uses the following configurable defaults:

- **Training**
  - **`--epochs`** (default: `50`) — number of training epochs.
  - **`--batch-size`** (default: `128`) — samples per training batch.
  - **`--lr`** (default: `3e-4`) — AdamW learning rate.
  - **`--wd`** (default: `1e-4`) — AdamW weight decay.
  - **`--workers`** (default: `4`) — dataloader worker processes.
- **Numerical stability**
  - **`eps`** (default: `1e-6`) — normalization clamp used in `MarginalNet.safe_softmax(...)`.
- **Model**
  - **`hidden`** (default: `256`) — channel width of MarginalNet.
- **Optimization**
  - **`grad_clip`** (default: `1.0`) — max grad-norm (`clip_grad_norm_`).
- **Loss weights / targets**
  - **`w_cdf`** (default: `10.0`) — multiplier on the image-resolution PDF L1 loss term.
- **Smoothing / stabilizers**
  - **`alpha0`** (default: `0.0`) — initial uniform-mix factor for predicted PDFs.
  - **`alpha_decay_steps`** (default: `2000`) — steps over which `alpha` decays linearly to `0`.

### Hardcoded training constants

These training settings are fixed in `train_scripts/marginalnet_full_dataset/trainer.py`:

- **Train/val split**: `0.9 / 0.1`.
- **Attention processing**: downsample attention maps to `(24, 24)` via `adaptive_avg_pool2d(...)` for supervision and visualization.
- **Optimizer**: `torch.optim.AdamW` (no LR scheduler).
- **Mixed precision**: AMP autocast + `GradScaler` when running on CUDA.
- **Loss**: image-resolution PDF L1 loss, scaled by `w_cdf`.


## Qualitative Results

<img width="550" height="663" alt="Qualitative Result 1" src="https://github.com/user-attachments/assets/0cc1b50f-7e6f-4ffc-aaea-6c8e064ea48d" />

<img width="550" height="663" alt="Qualitative Result 2" src="https://github.com/user-attachments/assets/7df7a6ae-e35a-4524-874b-5547a7af4e2f" />

## Colab Demo

- Run the setup cell (installs dependencies).
- Open the printed Cloudflare URL to launch the demo; it will initialize model downloads and warm up.
- Upload an image, enter a question, and click **Generate**.
- The demo uses a quantized LLaVA model to run on free Colab GPUs.
- Expected setup and model download time: ~10 minutes.
- The webiste will appear in the colab cell. 
  
```
Running in web server mode...
IN_COLAB: True, CLOUDFLARE_AVAILABLE: True
Colab detected - using Cloudflare tunnel. Please use Cloudflare URL
URL: https://omaha-governor-high-sunrise.trycloudflare.com
Local: http://localhost:5000
 * Serving Flask app 'run'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://172.28.0.12:5000

```

<img width="1350" height="920" alt="Colab Demo" src="https://github.com/user-attachments/assets/ceab3038-b9fd-4220-b192-0873a0eb6947" />

## Contact

For questions or support, please contact:
- Dwip Dalal: dwip2@illinois.edu

## BibTeX

If you find our work or code useful, please cite:

```bibtex
@article{dalal2025constructive,
  title={Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping},
  author={Dalal, Dwip and Vashishtha, Gautam and Mishra, Utkarsh and Kim, Jeonghwan and Kanda, Madhav and Ha, Hyeonjeong and Lazebnik, Svetlana and Ji, Heng and Jain, Unnat},
  journal={arXiv preprint arXiv:2510.09741},
  year={2025}
}
```

## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Qwen](https://github.com/QwenLM)
- [Mllm knows where to look](https://github.com/saccharomycetes/mllms_know) 

## License

This project is licensed under the MIT License - see the LICENSE file for details.
