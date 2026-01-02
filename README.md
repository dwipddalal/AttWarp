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

Create the environment using the provided YAML file:

```bash
conda env create -f attwarp.yaml
```

## Quick Setup Check

```bash
cd /path/to/AttWarp
python "Attention Guided Warping"/main_textvqa.py
python Experiments/TextVQA/evaluate_textvqa_accuracy.py
```

If your setup is correct, the above commands will return the number reported in Table 1 of the paper.

## Qualitative Results

<img width="550" height="663" alt="Qualitative Result 1" src="https://github.com/user-attachments/assets/0cc1b50f-7e6f-4ffc-aaea-6c8e064ea48d" />

<img width="550" height="663" alt="Qualitative Result 2" src="https://github.com/user-attachments/assets/7df7a6ae-e35a-4524-874b-5547a7af4e2f" />

## Colab Demo

- Run the setup cell (installs dependencies).
- Open the printed Cloudflare URL to launch the demo; it will initialize model downloads and warm up.
- Upload an image, enter a question, and click **Generate**.
- The demo uses a quantized LLaVA model to run on free Colab GPUs.
- Expected setup and model download time: ~10 minutes.

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
