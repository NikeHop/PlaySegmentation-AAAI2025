"""Merge the individual segmentations into a single annotation file."""

import argparse
import os
import glob
import random
import pickle
import wandb
import yaml

from collections import defaultdict

import numpy as np


def merge_segmentations_calvin(config: dict) -> None:
    """
    Merge segmentations from multiple files and update the partial annotation.

    Args:
        config (dict): Configuration parameters for the merging process.

    Returns:
        None
    """
    # Load data: full_annotation
    annotation_directory = os.path.join(
        config["dataset_directory"], "training", "lang_annotations"
    )

    full_annotation = np.load(
        os.path.join(annotation_directory, "auto_lang_ann.npy"), allow_pickle=True
    ).item()

    # Load data: row2task, instruction2emb, task2instructions
    frame2task = {}
    for frame, task in zip(
        full_annotation["info"]["indx"], full_annotation["language"]["task"]
    ):
        frame2task[frame] = task

    row2task = {}
    frame2row = full_annotation["language"]["task_frames2row"]
    for frame, row in frame2row.items():
        row2task[row] = frame2task[frame]

    instruction2emb = {}
    for ann, emb in zip(
        full_annotation["language"]["ann"], full_annotation["language"]["emb"]
    ):
        instruction2emb[ann] = emb

    task2instructions = defaultdict(list)
    for task, instruction in zip(
        full_annotation["language"]["task"], full_annotation["language"]["ann"]
    ):
        task2instructions[task].append(instruction)

    # Load partial annotation
    partial_annotation = np.load(
        os.path.join(annotation_directory, f"auto_lang_ann_{config['percent']}.npy"),
        allow_pickle=True,
    ).item()

    # Aggregate all segmentation files
    segmentation_files = glob.glob(f"{config['segmentation_directory']}/*.pkl")
    wandb.log({"#episodes2merge": len(segmentation_files)})

    segmentation = {"segmentation": [], "labels": [], "label_probs": []}
    for segmentation_file in segmentation_files:
        with open(segmentation_file, "rb") as file:
            episode_segmentation = pickle.load(file)

        segmentation["segmentation"] += episode_segmentation["segmentation"]
        segmentation["labels"] += episode_segmentation["labels"]
        segmentation["label_probs"] += episode_segmentation["label_probs"]

    # Compute some statistics about the dataset
    total_n_samples = 0
    samples_above_threshold = 0
    label_distribution = defaultdict(int)
    label_distribution_threshold = defaultdict(int)
    for frames, label, label_prob in zip(
        segmentation["segmentation"],
        segmentation["labels"],
        segmentation["label_probs"],
    ):
        total_n_samples += 1
        if label_prob > config["threshold"]:
            samples_above_threshold += 1
            label_distribution_threshold[label] += 1
        label_distribution[label] += 1

    wandb.log({"#samples": total_n_samples})
    wandb.log({"#sample_over_threshold": samples_above_threshold})

    for cl, n_samples in label_distribution.items():
        task = row2task[cl]
        wandb.log({f"#{task}": n_samples})

    for cl, n_samples in label_distribution_threshold.items():
        task = row2task[cl]
        wandb.log({f"#{task}_over_threshold": n_samples})

    # Add labelled segments to annotated dataset
    partial_annotation["language"]["emb"] = [
        elem.reshape(1, -1) for elem in list(partial_annotation["language"]["emb"])
    ]

    for frames, label, prob in zip(
        segmentation["segmentation"],
        segmentation["labels"],
        segmentation["label_probs"],
    ):
        if prob < config["threshold"]:
            continue

        task = row2task[label]
        ann = random.sample(task2instructions[task], k=1)[0]
        emb = instruction2emb[ann]

        partial_annotation["info"]["indx"].append(frames)
        partial_annotation["language"]["ann"].append(ann)
        partial_annotation["language"]["emb"].append(emb)
        partial_annotation["language"]["task"].append(task)

    partial_annotation["language"]["emb"] = np.concatenate(
        partial_annotation["language"]["emb"], axis=0
    )
    filepath = os.path.join(
        annotation_directory,
        f"auto_lang_ann_{config['percent']}_ps_{config['threshold']}.npy",
    )
    np.save(filepath, partial_annotation, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge the individual segmentations of CALVIN via Play Segmetation into a single annotation file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--segmentation_directory",
        type=str,
        required=True,
        help="Location of the segmentation files.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Integrate CLI arguments
    config["segmentation_directory"] = args.segmentation_directory

    # Initialize Wandb
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=["Pretraining"] + config["logging"]["tags"],
        name=f"Pretraining_{config['logging']['experiment_name']}",
        dir="../results",
    )

    wandb.config.update(config)

    merge_segmentations_calvin(config)
