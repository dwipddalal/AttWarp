# Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping


[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/6874350)
[![Project](https://img.shields.io/badge/Project-Page-black)](https://dwipddalal.github.io/Attwarp/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18GHH-SXyvfvrDUiZiuqaBuF7Ws2nierM#scrollTo=VWZ9cGqnkkW9)

## 📝 Introduction

This repository contains the code, model weights, and environment setup used in the paper - **Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping**.


<img width="1683" height="609" alt="image" src="https://github.com/user-attachments/assets/7ea03106-4615-4271-b931-e803be272f28" />



## MLLM Models 

The code and environment setup for MLLM models used in this work can be found here:

LLaVA: [Official code](https://github.com/haotian-liu/LLaVA)

Qwen: [Official code](https://github.com/QwenLM/Qwen3-VL)

## Attention Map Extraction 
We will soon release our attention map extraction code, but till then you can use: https://github.com/saccharomycetes/mllms_know code for attention map extraction.

### Quick Setup Check

```bash
cd AttWarp
python Attention-Guide Warping/main_textvqa.py
python Experiments/TextVQA/evaluate_textvqa_accuracy.py
```
If your setup is correct the above code will return the number reported in table 1

## ✅ TODO List

> Upcoming updates tracker .  

---

### AttWarp Map Extraction Code
Implementation of attention extraction for:
- [ ] LLaVA
- [ ] Qwen
- [ ] InternVL3
- [ ] InstructBLIP
- [x] Stable Diffusion

---

### Open-Vocabulary Object Detection with AttWarp
- [ ] Lisa-AttWarp code
- [ ] Inverse warping code
- [ ] Eval scripts

---

### Attention Redistribution Code
- [x] Calculation
- [x] Implementation
- [x] Metrics

---

### AttWarp-Distill Code
- [ ] Model-architecture
- [ ] Model weights 
- [ ] Evaluation and execution scripts.

---

### AttWarp-Chains Code
- [ ] Implementation Code 
- [ ] Termination Code



## 📧 Contact

For questions or support, please contact:
- Dwip Dalal: dwip2@illinois.edu

## Bibtex
If you find our work / code useful please cite:
```
@article{dalal2025constructive,
  title={Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping},
  author={Dalal, Dwip and Vashishtha, Gautam and Mishra, Utkarsh and Kim, Jeonghwan and Kanda, Madhav and Ha, Hyeonjeong and Lazebnik, Svetlana and Ji, Heng and Jain, Unnat},
  journal={arXiv preprint arXiv:2510.09741},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
