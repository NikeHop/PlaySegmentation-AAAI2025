# 💪 Train Labelling Model (LM)

Train labelling models on annotated dataset for CALVIN and BabyAI

**Note:** Activate `play_segmentation` conda-environment.

## BabyAI
To train the labelling model on the BabyAI dataset, run:

```
python train.py --config ./configs/babyai.yaml
```

**Compute Requirements**:
With the settings in `babyai.yaml`: ~1hr A100 40GB.


## CALVIN

To train the labelling model on the CALVIN dataset, run:

```
python train.py --config ./configs/calvin.yaml
```

**Compute Requirements**:
With the settings in `calvin.yaml`: ~1.5hrs 4xA100 40GB.

## Postprocessing

Compute metrics regarding the confidence of the LM and visualize labelled examples:

```
python postprocess.py --config ./configs/postprocess_babyai.yaml --checkpoint PATH_TO_CHECKPOINT
```

```
python postprocess.py --config ./configs/postprocess_calvin.yaml --checkpoint PATH_TO_CHECKPOINT
```

The examples can be found in the `./videos` folder.


## Pretrained Models:

- [BabyAI](https://drive.google.com/file/d/1vl_yKAKi-PWXsbKVOR8HR0K3PIcW9nFv/view?usp=sharing)
- [CALVIN](https://drive.google.com/file/d/1fZ5pigISKSsRu0vRfDltuHHG3OCkT_dO/view?usp=sharing)


## Acknowledgements

The ID3 model is taken from  [SlowFast](https://github.com/facebookresearch/SlowFast) (Copyright 2019, Facebook, Inc); Apache-2.0 license.
