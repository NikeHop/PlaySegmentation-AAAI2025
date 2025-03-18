# 💪📖 Training policies

Train policies to control agents in the BabyAI & CALVIN environment.

## Prerequisites

- You must have generated the necessary datasets.
    - For BabyAI see [here](../../play_segmentation/data/babyai/README.md)
    - For CALVIN see [here](../../play_segmentation/data/calvin/README.md)

## Training

### BabyAI

**Compute Requirements:** ~10min A100 40GB (not necessary)

To train a policy in the BabyAI environment run:

```
python imitation.py --config ./configs/imitation_babyai.yaml --dataset-file NAME_DATAFILE.
```

Here NAME_DATAFILE is the name of the training dataset (e.g. `babyai_go_to_10000_single_training.pkl`).
The policy is evaluated after training.

### MCIL

**Compute Requirements:** ~15hrs A100 40GB

To train a policy on the CALVIN benchmark, run:

```
python mcil.py --config ./configs/mcil_calvin.yaml
```

To change the dataset the policy for CALVIN is trained on change the `episode_filename` in the `mcil_calvin.yaml` file.


To evaluate the trained MCIL policy, see the repo [here](https://github.com/NikeHop/mcil_evaluation_calvin.git).
