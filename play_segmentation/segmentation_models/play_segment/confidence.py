""" Determine the confidence accuracy relation of the PlaySegmentation model. """

import argparse

import torch
import torch.nn.functional as F
import tqdm
import wandb
import yaml

from play_segmentation.segmentation_models.play_segment.data import (
    get_data_play_segmentation,
)
from play_segmentation.segmentation_models.play_segment.trainer import PlaySegmentation


def accuracy_confidence(config: dict) -> None:
    """
    Compute the accuracy-confidence relation for Play Segmentation.

    Args:
        config (dict): Configuration parameters for the accuracy-confidence computation.

    Returns:
        None
    """

    # Load model
    model = PlaySegmentation.load_from_checkpoint(config["checkpoint"])
    model.eval()

    # Load validation data
    _, validation_dataloader = get_data_play_segmentation(config["data"])

    # Initialize threshold to statistics dictionary
    threshold2statistics = {
        threshold: {"n_predictions": 0, "n_correct": 0}
        for threshold in config["thresholds"]
    }

    # For each validation sample determine confidence and accuracy
    total_n_predictions = 0
    for sample in tqdm.tqdm(validation_dataloader):
        # Send to Device
        for key, value in sample.items():
            sample[key] = value.cuda() if isinstance(value, torch.Tensor) else value

        # Compute Predictions
        seg_probs, cls_probs = model.model(sample)
        cls_probs = F.softmax(cls_probs, dim=-1)
        pred_cls_probs, pred_cls_labels = cls_probs.max(dim=-1)
        gt_labels = [elem for elem in sample["class_labels"] if elem != -1]

        # Update class statistics
        for pred_cls_prob, pred_cls_label, gt_label in zip(
            pred_cls_probs, pred_cls_labels, gt_labels
        ):
            total_n_predictions += 1
            for threshold, statistics in threshold2statistics.items():
                if pred_cls_prob >= threshold:
                    statistics["n_predictions"] += 1
                    if pred_cls_label == gt_label:
                        statistics["n_correct"] += 1

    # Log the Accuracy Confidence relation
    for threshold, statistics in sorted(
        threshold2statistics.items(), key=lambda x: x[0]
    ):
        accuracy = (
            statistics["n_correct"] / statistics["n_predictions"]
            if statistics["n_predictions"] > 0
            else 0
        )
        coverage = (
            statistics["n_predictions"] / total_n_predictions
            if total_n_predictions > 0
            else 0
        )
        wandb.log({"Accuracy": accuracy, "Confidence": threshold, "Coverage": coverage})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Determine the confidence accuracy relation of the model."
    )
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load TriDet model from.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Integrate CL arguments into config
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint

    # WandDB initialization
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=["Segmentation"] + config["logging"]["tags"],
        name=f"Segmentation_{config['logging']['experiment_name']}",
        dir="../results",
    )
    wandb.config.update(config)

    accuracy_confidence(config)
