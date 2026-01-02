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


## Qualitative Results

<img width="550" height="663" alt="image" src="https://github.com/user-attachments/assets/0cc1b50f-7e6f-4ffc-aaea-6c8e064ea48d" />

<img width="550" height="663" alt="image" src="https://github.com/user-attachments/assets/7df7a6ae-e35a-4524-874b-5547a7af4e2f" />


## The Google Colab Demo

- Run the setup cell below (setups up dependencies).
- After running next cell Open the printed Cloudflare URL to open demo, initalizing model downloading and warm up.
- Upload an image, enter a question, click Generate.
- Note to make it work on the free gpus of google colab the llava model that we are using in it is quantized version. 
- Expected setup and model download time ~ 10 mins
- The website output looks like following:

<img width="1350" height="920" alt="image" src="https://github.com/user-attachments/assets/ceab3038-b9fd-4220-b192-0873a0eb6947" />



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
