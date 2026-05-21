# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCBSegClassNet is a TensorFlow-based deep learning project for PCB (Printed Circuit Board) component segmentation and classification. It uses the FICS PCB Image Collection (FPIC) dataset.

The two tasks are handled by separate model variants sharing the same encoder:
- **Segmentation**: `PCBSegNet` — segments all 25 component classes on a full PCB image
- **Classification**: `PCBClassNet` — classifies individual cropped component images

## Environment Setup

```bash
conda create -n pscn python=3.8
conda activate pscn
conda install pip
pip install -r requirements.txt
```

Key dependencies: `tensorflow-gpu==2.11`, `albumentations`, `pyyaml`, `tqdm`, `pandas`.

## Commands

All training commands must be run from the `src/` directory.

**Train segmentation** (100 epochs):
```bash
python train_segmentation.py -opt cfs/pscn_seg.yml -epoch 100
```

**Evaluate segmentation** (loads best checkpoint, skips training):
```bash
python train_segmentation.py -opt cfs/pscn_seg.yml -epoch 0
```

**Train classification** (100 epochs):
```bash
python train_classification.py -opt cfs/pscn_class.yml -epoch 100
```

**Evaluate classification**:
```bash
python train_classification.py -opt cfs/pscn_class.yml -epoch 0
```

**Data preparation** (run from `src/data/`):
```bash
# Create HSI+CLAHE images, masks, and classification crops
python create_mask.py -i ../../data/pcb_image/ -a ../../data/smd_annotation/ -id ../../data/segmentation/images -ad ../../data/segmentation/masks -cd ../../data/classification/images/

# Create patches (768px) and split train/test
python create_patches.py -i ../../data/segmentation/images/ -m ../../data/segmentation/masks -cd ../../data/classification/images/ -ps 768
```

## Architecture

### Encoder (shared by both tasks)
Built in `src/models/blocks.py`, the encoder has three stages:
1. **Learning Module** — three conv/depthwise-separable conv blocks with stride 2, producing feature maps at 3 scales (`learning_layer1`, `learning_layer2`, `learning_layer3`)
2. **Feature Extractor** — three `bottleneck_block` stages (MobileNetV2-style residual bottlenecks) followed by a `pyramid_pooling_block` (PSPNet-style)
3. **Fusion Module** — fuses the learning module output with the upsampled feature extractor output

### Segmentation Decoder (`get_decoder` in `blocks.py`)
- Applies `tem_block` (Texture Enhancement Module: channel attention + cosine-similarity-based spatial attention) to encoder output
- Two upsampling steps with skip connections from `learning_layer2` and `learning_layer1`
- Final `Conv2D(num_classes)` + softmax

### Classification Head (`get_classification` in `blocks.py`)
- `GlobalAveragePooling2D` on encoder output → `Dense(128, relu)` → `Dense(num_classes, softmax)`

### Loss
Segmentation uses **DISLoss** (`src/models/loss.py`): sum of Dice loss + Jaccard loss + SSIM loss. Classification uses standard `categorical_crossentropy`.

## Configuration

Training hyperparameters and data paths are controlled by YAML files in `src/cfs/`:
- `pscn_seg.yml` — segmentation config (25 classes, Adam lr=1e-4, batch=16, input 512×512)
- `pscn_class.yml` — classification config (25 classes, Adam lr=1e-4, batch=16, input 512×512)

Checkpoints are saved to `checkpoints/best_seg.h5` and `checkpoints/best_class.h5`. Logs go to `logs/app.log`.

## Data

25 PCB component classes: R, C, U, Q, J, L, RA, D, RN, TP, IC, P, CR, M, BTN, FB, CRA, SW, T, F, V, LED, S, QA, JP.

The segmentation masks use specific RGB color values per class (defined in `src/data/dataloader.py::color_values`). When modifying mask generation, ensure colors match this mapping exactly.

The FPIC dataset requires access codes from the dataset authors — it is not freely downloadable.
