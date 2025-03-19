# 🎭✂️ Play Segmentation - AAAI 2025

Codebase for the AAAI-2025 paper ["Data Augmentation for Instruction Following Policies via Trajectory Segmentation"](https://arxiv.org/abs/2503.01871)

[Project Page](https://nikehop.github.io/play_segmentation/)

## Prerequisites

- Anaconda3/Miniconda3

```
conda create -n play_segmentation python=3.11
conda activate play_segmentation
pip install -e .
```

## 🔍 Overview

Overview of the different components of Play Segmentation

1. First get the necessary Datasets BabyAI/CALVIN.
2. Train a Labelling Model on the annotated datasets.
3. Train Segmentation Models.
4. Train a policy.


## 🗂️ Datasets

### Calvin

To get the CALVIN dataset follow the instructions [here](./play_segmentation/data/calvin/README.md).


### BabyAI

To generate the segmented as well as unsegmented trajectories in the BabyAI environment follow the instructions [here](./play_segmentation/data/babyai/README.md).

## 📝 Labelling

### Train Labelling Model

**Note:** The labelling model also needs to be trained to obtain pretrained embeddings of videos for the segmentation methods TriDet and Play Segmentation.

To train the labelling model follow the instructions [here](./play_segmentation/labelling/labelling_training/README.md).

### Label Data

To apply the trained segmentation models to extract labelled segments from play trajectories, follow the instructions [here](./play_segmentation/labelling/labelling/README.md).

## ✂️ Segmentation

To train the different segmentation models follow the instructions [here](./play_segmentation/segmentation_models/README.md). At the moment the following segmentation models are implemented:

- Play Segmentation (Ours)
- UnLoc (https://arxiv.org/abs/2308.11062)
- TriDet (https://arxiv.org/abs/2303.07347) / [Code](https://github.com/dingfengshi/TriDet)


## 💪 Policy Training

To train an imitation learning policy for BabyAI and a policy via MCIL for CALVIN see [here](./play_segmentation/policy/README.md).



## Acknowledgements

- [BabyAI](https://github.com/mila-iqia/babyai), BSD 3-Clause License
- [CALVIN](https://github.com/mees/calvin), MIT license
- [TriDet](https://github.com/dingfengshi/TriDet), MIT license
