<img src = "data/sample/506.jpg" alt = "origin image" width = "256" height = "256"/>
<img src = "data/sample/506_upcaled.png" alt = "origin image" width = "512" height = "512"/>


📌 **Image Super-Resolution ISR**

An implementation of Residual Dense Network (RDN) for single image super-resolution using PyTorch.

This project aims to reconstruct high-resolution images from low-resolution inputs, achieving high PSNR and SSIM performance.
## Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)

## Overview
Image super-resolution is a computer vision task that aims to reconstruct a high-resolution (HR) image from its low-resolution (LR) counterpart. This project implements the **Residual Dense Network (RDN)** architecture in PyTorch, a state-of-the-art deep learning model designed for single image super-resolution.

RDN leverages dense connections and residual learning to effectively extract and fuse hierarchical features, leading to superior reconstruction quality. With this implementation, users can train the model from scratch on their own datasets or utilize pre-trained weights for fast inference. 

The project supports multiple scaling factors (e.g., ×2, ×3, ×4) and has been evaluated on standard benchmarks such as DIV2K, Set5, and Set14, achieving competitive PSNR and SSIM scores.
This implementation is inspired by the original paper: [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797) (Zhang et al. 2018).

<img width="658" height="188" alt="image" src="https://github.com/user-attachments/assets/72efa6e5-d3c7-4040-983a-3045a7b98f6e" />

<img width="481" height="162" alt="image" src="https://github.com/user-attachments/assets/59058f23-34b0-420d-940b-5fe79053557a" />

## Project Structure
```
Directory structure:
└── cor1211-image_super_resolution/
    ├── README.md
    ├── environment.yml
    └── RDN/
        ├── imgNet_dataset.py
        ├── RDN.py
        ├── test_RDN.py
        └── train_RDN.py
```

## Features
- **State-of-the-art architecture** – Implements the [Residual Dense Network (RDN)](https://arxiv.org/abs/1802.08797) for single image super-resolution.
- **Multi-scale support** – Train and test with scaling factors ×2, ×3, ×4.
- **Pre-trained models** – Ready-to-use weights for fast inference.
- **Custom dataset support** – Easily train on your own datasets in addition to standard benchmarks such as DIV2K, Set5, and Set14.
- **Data preprocessing utilities** – Includes scripts for creating low-resolution images (`create_LR.py`) and cropping datasets (`crop_image.py`).
- **Evaluation metrics** – PSNR and SSIM evaluation for quantitative performance comparison.
- **GPU acceleration** – Optimized for CUDA-enabled devices to speed up training and inference.
## Installation
- Install ISR from the GitHub source:
  ```
  git clone https://github.com/cor1211/image_super_resolution.git 
  conda env create -f environment.yml # Need to install anaconda/miniconda before
  ```
## Using
**Prediction**
```
cd RDN/
python test_RDN.py --image_path "your_image_path"
```

