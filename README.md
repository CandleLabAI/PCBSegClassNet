PCBSegClassNet - A Light-weight Network for Segmentation and Classification of PCB Component
==============================

## Overview:

This repository contains the source code of our paper, PCBSegClassNet (Under review for publication in <a href="https://www.sciencedirect.com/journal/expert-systems-with-applications">Expert Systems With Applications</a>).

We present PCBSegClassNet, a novel deep neural network for PCB component segmentation and classification. We perform segmentation on the whole PCB image and classification on the individual component images. We design a novel two-branch network for achieving high segmentation and classification performance.

## Flow of network

<img src="reports/figures/flow.png" width=100%>

<hr>

Project Organization
------------
    ├── LICENSE                         <- The LICENSE for developers using this project.
    ├── README.md                       <- The top-level README for developers using this project.
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`.
    |── reports                         <- The directory containing metadata used for repo.
    ├── checkpoints                     <- Directory where best models will be saved.
    ├── src                             <- Source code for use in this project.
    │   ├── data                        
    │   │   └── dataloader.py           <- Source code for generating data loader.
    |   ├── cfs
    │   │   └── pscn_class.yml          <- basic configurations for classification training of PCBSegClassNet model.
    │   │   └── pscn_seg.yml            <- basic configurations for segmentation training of PCBSegClassNet model.
    │   ├── models
    │   │   ├── blocks.py               <- Source code for the individual blocks used for creating network of PCBSegClassNet.
    │   │   └── network.py              <- Source code for the PCBSegClassNet network for both segmentation and classification task.
    |   |   └── loss.py                 <- Source code for proposed DIS Loss and other metrics used for evaluation.
    │   |── utils
    |   |   └── options.py              <- Source code for parsing yaml file.
    |   └── train_classification.py     <- Source code for training and testing of classification network.
    |   └── train_segmentation.py       <- Source code for training and testing of segmentation network.
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────
------------

<hr>

## Network Architecture

<img src="reports/figures/network.png" width=100%>

<hr>

## Sample Results

<img src="reports/figures/sample_results.png" width=100%>

<hr>

## Get Started

Dependencies:

```bash
pip install -r requirements.txt
```
### (Optional) Conda Environment Configuration

First, create a conda environment
```bash
conda create -n pscn # python=3
source activate pscn
pip install -r requirements.txt
```

### Dataset

We have used FICS PCB Image Collection (FPIC) dataset which can be downloaded from <a href="https://www.trust-hub.org/#/data/pcb-images">here</a>. `pcb_image.zip` and `smd_annotation.zip` needs to be downloaded, unzipped and placed under `data/` directory.

NOTE: YOU NEED TO REQUEST FOR ACCESS CODES FROM AUTHOR TO DOWNLOAD THIS DATASET.

Data Structure after completing above steps
------------
    ├── smd_annotation                  <- Directory containing annotations in csv format.
    │   └── pcb_0f_cc_11_smd.csv
    │   └── pcb_1f_cc_2_smd.csv
    │   └── ...
    ├── pcb_image                       <- Directory containing input images.
    │   └── pcb_0b_cc_11.png
    │   └── pcb_0f_cc_11.png
    │   └── ...
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────
------------

### Data preparation

To prepare HSI + Clahe images, and Masks for segmentation and crops for  classification, run following code from `src/data` directory

```bash
python create_mask.py -i ../../data/pcb_image/ -a ../../data/smd_annotation/ -id ../../data/segmentation/images -ad ../../data/segmentation/masks -cd ../../data/classification/images/
```

To create patches and split data into train/test run following code from `src/data` directory. Here we choosed patch size as 768

```bash
python create_patches.py -i ../../data/segmentation/images/ -m ../../data/segmentation/masks -cd ../../data/classification/images/ -ps 768
```

Data Structure after completing above steps
------------
    ├──data
    │  ├── sementation
    │  │   ├── train
    │  │   │   ├── images
    │  │   │   │   └── image_0.png
    │  │   │   │   └── image_1.png
    │  │   │   │   └── ...
    │  │   │   ├── masks
    │  │   │   │   └── image_0.png
    │  │   │   │   └── image_1.png
    │  │   │   │   └── ...
    │  │   ├── test
    │  │   │   ├── images
    │  │   │   │   └── image_0.png
    │  │   │   │   └── image_1.png
    │  │   │   │   └── ...
    │  │   │   ├── masks
    │  │   │   │   └── image_0.png
    │  │   │   │   └── image_1.png
    │  │   │   │   └── ...
    │  ├── classification
    │  │   ├── train
    │  │   │   ├── BTN
    │  │   │   │   └── image_0.png
    │  │   │   │   └── image_1.png
    │  │   │   │   └── ...
    │  │   │   ├── C
    │  │   │   │   └── image_0.png
    │  │   │   │   └── image_1.png
    │  │   │   │   └── ...
    │  │   │   ...
    │  │   ├── val
    │  │   │   ├── BTN
    │  │   │   │   └── image_0.png
    │  │   │   │   └── image_1.png
    │  │   │   │   └── ...
    │  │   │   ├── C
    │  │   │   │   └── image_0.png
    │  │   │   │   └── image_1.png
    │  │   │   │   └── ...
    │  │   │   ...
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────
------------

### Training Segmentation model

change the hyperparameters and configuration parameters according to need in ```cfs/pscn_seg.yml```.

To train pscn, Run following command from ```/src``` directory.

```bash
python train_segmentation.py -opt cfs/pscn_seg.yml -epoch 100
``` 
Above command will train segmentation model for 100 epochs with given configuration.

The trained checkpoint for model training will be saved in ```/checkpoints/best_seg.h5```

### Testing Segmentatin performance

To test pscn with trained model, Run following command from ```/src``` directory.

```bash
python train_segmentation.py -opt cfs/pscn_seg.yml -epoch 0
```

Above command will generate IoU Score, and DICE Score for segmentation output.

### Training Classification model

change the hyperparameters and configuration parameters according to need in ```cfs/pscn_class.yml```.

To train pscn, Run following command from ```/src``` directory.

```bash
python train_classification.py -opt cfs/pscn_class.yml -epoch 100
``` 
Above command will train classification model for 100 epochs with given configuration.

The trained checkpoint for model training will be saved in ```/checkpoints/best_class.h5```

### Testing Segmentatin performance

To test pscn with trained model, Run following command from ```/src``` directory.

```bash
python train_classification.py -opt cfs/pscn_class.yml -epoch 0
```

Above command will generate Accuracy, Precision, and Recall for classification output.

## Citation
```
@article {ref162,
	title = "PCBSegClassNet - A Light-weight Network for Segmentation and Classification of PCB Component",
	year = "2023",
	author = "Dhruv Makwana and R Sai Chandra Teja and Sparsh Mittal",
	journal = "Expert Systems With Applications",
}
```
## License
<hr>
CC BY-NC-ND 4.0