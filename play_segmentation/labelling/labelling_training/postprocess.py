"""Post-training analysis of I3D labelling model"""

import argparse
import os

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml

from einops import rearrange
from torchvision.io import write_video

from play_segmentation.data.babyai.utils import CL2INSTRUCTION_GOTOLOCAL, state2img
from play_segmentation.labelling.labelling_training.data import get_data
from play_segmentation.labelling.labelling_training.trainer import I3D


def postprocess(config: dict) -> None:
    """
    Postprocesses the predictions and computes statistics.

    Args:
        config (dict): Configuration parameters for postprocessing.

    Returns:
        None
    """
    # Create directory
    if config["visualize"]:
        os.makedirs(f"./videos/{config['data']['env']}", exist_ok=True)

    # Load data
    _, validation_data = get_data(config["data"])

    # Load model
    model = I3D.load_from_checkpoint(
        config["checkpoint"], map_lcoation=config["device"]
    )
    model.eval()

    # Prepare statistics
    thresholds = [0.05 * i for i in range(0, 20)]
    threshold2metrics = defaultdict(
        lambda: defaultdict(int)
    )  # Get the number of tp, fp, fn, tn for each threshold
    task2pn = defaultdict(
        lambda: defaultdict(list)
    )  # Get the average confidence of the fp (false positive), tp,  fn, tn

    total = 0
    for batch_id, (videos, video_lengths, states, labels) in enumerate(validation_data):
        # Compute predictions
        probs = model.model(videos.to(config["device"]))
        preds = probs.argmax(dim=-1)

        # Compute visualizations if needed
        if config["visualize"] and batch_id == 0:
            visualize_predictions(videos, video_lengths, states, preds, config)

        # Compute statistics
        probs = probs.max(dim=-1)[0]
        for predicted_label, label, prob in zip(preds, labels, probs):
            predicted_label = predicted_label.cpu().item()
            prob = prob.cpu().item()
            positive = predicted_label == label
            if positive:
                task2pn[predicted_label]["p"].append(prob)
            else:
                task2pn[predicted_label]["n"].append(prob)

            for threshold in thresholds:
                if prob > threshold:
                    threshold2metrics[threshold]["total"] += 1
                    threshold2metrics[threshold]["correct"] += positive
            total += 1

    # Log statistics: Is the model confidence better for correct predictions?
    for task, pn in task2pn.items():
        if len(pn["p"]) > 0:
            p = sum(pn["p"]) / len(pn["p"])
        else:
            p = 0
        if len(pn["n"]) > 0:
            n = sum(pn["n"]) / len(pn["n"])
        else:
            n = 0

        wandb.log({f"task_{task}/p": p, f"task_{task}/n": n})
        print(
            {f"task_{task}/p": (p, len(pn["p"])), f"task_{task}/n": (n, len(pn["n"]))}
        )

    # Accuracy per confidence threshold:
    for threshold, metrics in threshold2metrics.items():
        wandb.log(
            {
                f"threshold_{threshold}_acceptance": {metrics["total"] / total},
                f"threshold_{threshold}_accuracy": {
                    metrics["correct"] / metrics["total"]
                },
            }
        )
        print(
            f"Threshold {threshold}: Acceptance: {metrics['total']/total} Accuracy: {metrics['correct']/metrics['total']}"
        )


def visualize_predictions(
    videos: torch.Tensor,
    video_lengths: torch.Tensor,
    states: torch.Tensor,
    preds: torch.Tensor,
    config: dict,
) -> None:
    """
    Visualizes the predictions by writing videos to disk.

    Args:
        videos (torch.Tensor): Input videos tensor of shape (batch_size, T, H, W, C).
        preds (torch.Tensor): Predictions tensor of shape (batch_size,).
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    if config["data"]["env"] == "babyai":
        visualize_predictions_babyai(states, video_lengths, preds, config)
    elif config["data"]["env"] == "calvin":
        visualize_predictions_calvin(videos, video_lengths, preds, config)
    else:
        raise NotImplementedError("This environment has not been implemented")


def visualize_predictions_babyai(
    states: torch.Tensor, video_lengths: torch.Tensor, preds: torch.Tensor, config: dict
) -> None:
    """
    Visualizes the predictions made by a model on BabyAI environment.

    Args:
        states (torch.Tensor): Tensor containing the states of the trajectories.
        video_lengths (torch.Tensor): Tensor containing the lengths of each video.
        preds (torch.Tensor): Tensor containing the predicted labels for each video.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    # Transform states to videos
    states = states.numpy()
    videos = []
    for i, traj in enumerate(states[: config["n_visualizations"]]):
        video = []
        for state in traj[: video_lengths[i]]:
            video.append(torch.from_numpy(state2img(state)))
        video = torch.stack(video, dim=0)
        videos.append(video)

    for i, video in enumerate(videos):
        inst = CL2INSTRUCTION_GOTOLOCAL[preds[i].cpu().item()]
        write_video(f"./videos/babyai/video_{i}_{inst}.mp4", video, fps=5)


def visualize_predictions_calvin(
    videos: torch.Tensor, video_lengths: torch.Tensor, preds: torch.Tensor, config: dict
):
    """
    Visualizes the predictions made by a model on a set of videos.

    Args:
        videos (torch.Tensor): A tensor containing the videos.
        video_lengths (torch.Tensor): A tensor containing the lengths of the videos.
        preds (torch.Tensor): A tensor containing the predicted labels.
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """
    filepath = os.path.join(
        config["data"]["dataset_directory"],
        "training",
        "lang_annotations",
        "auto_lang_ann.npy",
    )
    annotations = np.load(filepath, allow_pickle=True).item()

    frames2task = {
        frames: task
        for frames, task in zip(
            annotations["info"]["indx"], annotations["language"]["task"]
        )
    }
    row2task = {}
    for frames, row in annotations["language"]["task_frames2row"].items():
        task = frames2task[frames]
        row2task[row] = task
    print(row2task)

    for i, (video, video_length, preds) in enumerate(zip(videos, video_lengths, preds)):
        label = preds.cpu().item()
        task = row2task[label]
        video = video[:video_length]
        write_video(f"./videos/calvin/video_{i}_{task}.mp4", video, fps=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="path to experiment config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="path to model checkpoint"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    # Integrate CL arguments into config
    if args.checkpoint is not None:
        config["checkpoint"] = args.checkpoint

    # Setup wandb project
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=["Pretraining"] + config["logging"]["tags"],
        name=f"Pretraining_{config['logging']['experiment_name']}",
        dir="../results",
    )
    wandb.config.update(config)

    with torch.no_grad():
        postprocess(config)
