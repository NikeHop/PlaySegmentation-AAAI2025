# 💪✂️ Train segmentation models

Train one of the following segmentation model:

1. [UnLoc](https://arxiv.org/abs/2308.11062)
2. [TriDet](https://arxiv.org/abs/2309.05590)
3. Play Segmentation (Ours)


**Requirements:**
- Generated/Downloaded datasets for BabyAI and CALVIN (see [here](/play_segmentation/data/babyai/README.md) & [here](/play_segmentation/data/calvin/README.md)).
- TriDet and Play Segmentation need pretrained I3D model (see [here](/play_segmentation/labelling/labelling_training/README.md)).

**Note:** Activate the `play_segmentation` conda-environment

## Training

### UnLoc

#### BabyAI

**Compute Requirements:** ~8hr on 1 x A100 40GB

```
python train.py --config ./configs/train_unloc_babyai.yaml
```

#### CALVIN

**Compute Requirements:** ~5hr on 2 x A100 40

```
python train.py --config ./configs/train_unloc_calvin.yaml
```


### TriDet

#### BabyAI

**Compute Requirements:** ~1.5hr on 1 x A100 40GB

```
python train.py --config ./configs/train_tridet_babyai.yaml --id3_checkkpoint PATH_TO_I3D_CHECKPOINT
```

#### CALVIN

**Compute Requirements:** ~1hr on 4 x A100 40GB

```
python train.py --config ./configs/train_tridet_calvin.yaml --id3_checkkpoint PATH_TO_I3D_CHECKPOINT
```

To determine the relation between confidence and accuracy run:

```
python tridet/confidence.py --config ./configs/tridet_confidence_calvin.yaml --checkpoint PATH_TO_TRIDET_CHECKPOINT
```

### Play Segmentation

#### BabyAI

**Compute Requirements:** ~30min on 1 x A100 40GB
```
python train.py --config ./configs/train_ps_babyai.yaml --id3_checkkpoint PATH_TO_I3D_CHECKPOINT
```

#### CALVIN

**Compute Requirements:** ~8hr on 4 x A100 40GB

```
python train.py --config ./configs/train_ps_calvin.yaml --id3_checkkpoint PATH_TO_I3D_CHECKPOINT
```

To determine the relation between confidence and accuracy run:

```
python play_segment/confidence.py --config ./configs/ps_confidence_calvin.yaml --checkpoint PATH_TO_PS_CHECKPOINT
```


## Pretrained Models
You can download pretrained models:

BabyAI:

- [TriDet](https://drive.google.com/file/d/1FzoioUfD4mvnCg4MhKHIIHrArRa_YpH3/view?usp=sharing)
- [UnLoc](https://drive.google.com/file/d/1xMYSnySV3OI23AqRWgUVU9wuIkQFDpKL/view?usp=sharing)
- [Play Segmentation](https://drive.google.com/file/d/1K9kBrTwLEs5ZYo7WDmeI8cqmKhPJVKNM/view?usp=sharing)

CALVIN:

- [TriDet](https://drive.google.com/file/d/1ytaUZllwLZGHZIj4j_7hS6XjAHSRRodT/view?usp=sharing)
- [UnLoc](https://drive.google.com/file/d/1Di55nDN6VOpbY7OxTXiKs6QWRl1akp0Y/view?usp=sharing)
- [Play Segmentation](https://drive.google.com/file/d/11IllZ_yGqwZeXP7LfaeaT44DxhPJE0N8/view?usp=sharing)
