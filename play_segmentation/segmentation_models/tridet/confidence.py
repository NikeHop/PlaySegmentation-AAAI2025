"""Determine the confidence accuracy relation of the TriDet model."""

import argparse

import torch
import tqdm
import wandb
import yaml

from play_segmentation.segmentation_models.tridet.data import get_data_tridet
from play_segmentation.segmentation_models.tridet.trainer import TriDet


def accuracy_confidence(config: dict) -> None:
    """
    Compute the accuracy-confidence relation for a trained TriDet model.

    Args:
        config (dict): Configuration dictionary containing the following keys:
            - "checkpoint" (str): Path to the model checkpoint.
            - "data" (dict): Configuration dictionary for loading validation data.
            - "thresholds" (list): List of confidence thresholds to evaluate.

    Returns:
        None
    """

    # Load model
    model = TriDet.load_from_checkpoint(config["checkpoint"])
    model.eval()

    # Load validation data
    _, validation_dataloader = get_data_tridet(config["data"])

    # Initialize threshold to statistics dictionary
    threshold2statistics = {
        threshold: {"n_predictions": 0, "n_correct": 0}
        for threshold in config["thresholds"]
    }
    total_n_predictions = 0

    # For each validation sample determine confidence and accuracy
    for sample in tqdm.tqdm(validation_dataloader):
        # Send to GPU
        sample = [
            elem.cuda() if isinstance(elem, torch.Tensor) else elem for elem in sample
        ]
        results = model.model(*sample)
        for result, gt_segment, gt_label in zip(results, sample[2], sample[3]):
            total_n_predictions += 1

            # Determine top segment
            max_value, max_index = result["scores"].max(dim=0)
            max_value = max_value.item()
            max_index = max_index.item()

            # Determine predicted class
            pred_label = result["labels"][max_index].cpu()

            # Compute threshold specific statistics
            for threshold, statistics in threshold2statistics.items():
                if max_value >= threshold:
                    statistics["n_predictions"] += 1
                    if pred_label == gt_label:
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
