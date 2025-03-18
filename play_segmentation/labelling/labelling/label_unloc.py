"""Create a labelled dataset from the UnLoc model."""

import argparse
import logging
import os
import pickle
import random

from collections import defaultdict
from typing import Any

import blosc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
import yaml

from einops import repeat, rearrange
from torch.utils.data import Dataset
from torchvision.io import write_video

from play_segmentation.data.babyai.utils import (
    CL2INSTRUCTION_GOTOLOCAL,
    CL2INSTRUCTION_TYPE_GOTOLOCAL,
)
from play_segmentation.labelling.labelling.utils import create_babyai_video, load_data
from play_segmentation.segmentation_models.unLoc.data import transform2clip
from play_segmentation.segmentation_models.unLoc.trainer import UnLoc
from play_segmentation.utils.vis import visualize_video


class LabellingDatasetUnLocBabyAI(Dataset):
    """
    Dataset class for labeling in the BabyAI environment using UnLoc.

    Args:
        dataset (dict): The dataset containing images, actions, directions, rewards, and states.
        max_window (int): The maximum window size for sampling segments.
        instruction_embeddings (torch.Tensor): Embeddings of instructions.
        inst2cl (dict): Mapping of instructions to classes.
        device (torch.device): The device to use for computation.

    Attributes:
        dataset (dict): The dataset containing images, actions, directions, rewards, and states.
        max_window (int): The maximum window size for sampling segments.
        instruction_embeddings (torch.Tensor): Embeddings of instructions.
        inst2cl (dict): Mapping of instructions to classes.
        n_instructions (int): Number of instructions.
        length (int): Length of the dataset.
        device (torch.device): The device to use for computation.
    """

    def __init__(
        self,
        dataset: dict,
        max_window: int,
        instruction_embeddings: torch.Tensor,
        inst2cl: dict,
        device: torch.device,
    ):
        self.dataset = dataset
        self.max_window = max_window
        n_instructions, D = instruction_embeddings.shape
        self.instruction_embeddings = torch.cat(
            [instruction_embeddings, torch.zeros(1, D)], dim=0
        )
        self.inst2cl = inst2cl
        self.n_instructions = n_instructions
        self.length = len(dataset["images"])
        self.device = device

    def sample_segment(self) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any]]:
        """
        Sample a segment from the dataset.

        Returns:
            batch (dict): A dictionary containing the sampled segment data.
            video (torch.Tensor): The unsegmented trajectory video.
            traj (dict): A dictionary containing the trajectory data.
        """
        index = random.randint(0, self.length - 1)

        # Sample a random unsegmented trajectory from the dataset
        unseg_traj_clip = self.dataset["clip_images"][index]
        unseg_traj_clip = [
            torch.tensor(blosc.unpack_array(segment)[:-1], dtype=torch.float)
            for segment in unseg_traj_clip[:-1]
        ] + [torch.tensor(blosc.unpack_array(unseg_traj_clip[-1]), dtype=torch.float)]
        unseg_traj_clip = torch.cat(unseg_traj_clip, dim=0)
        length = unseg_traj_clip.shape[0]

        # Video
        unseg_traj = self.dataset["images"][index]
        unseg_traj = [
            torch.tensor(blosc.unpack_array(segment)[:-1], dtype=torch.float)
            for segment in unseg_traj[:-1]
        ] + [torch.tensor(blosc.unpack_array(unseg_traj[-1]), dtype=torch.float)]
        unseg_traj = torch.cat(unseg_traj, dim=0)

        # Sample a random start time and window
        start_time = random.randint(0, length - 2)
        window = random.randint(1, min(self.max_window, length - start_time - 1))
        clip_video = unseg_traj_clip[start_time : start_time + window]
        video = unseg_traj[start_time : start_time + window]

        # clip_seqs, obs_length, img_length
        clip_video = repeat(clip_video, "t d -> t c d", c=self.n_instructions + 1)
        clip_video = torch.cat(
            [clip_video, self.instruction_embeddings.unsqueeze(0)], dim=0
        ).to(self.device)

        obs_length = torch.tensor([window + 1], dtype=torch.long).to(self.device)
        img_obs_length = torch.tensor([window], dtype=torch.long).to(self.device)

        clip_video = rearrange(clip_video, "t c d -> c t d").unsqueeze(0)
        batch = {
            "clip_seq": clip_video,
            "obs_lengths": obs_length,
            "img_obs_lengths": img_obs_length,
        }

        # Collect the data necessary to form a new sample
        traj = {}
        traj["obs"] = unseg_traj[start_time : start_time + window + 1]

        actions = [a for action in self.dataset["actions"][index] for a in action]
        traj["actions"] = actions[start_time : start_time + window]

        directions = [
            d for direction in self.dataset["directions"][index] for d in direction
        ]
        traj["directions"] = directions[start_time : start_time + window]

        rewards = [r for reward in self.dataset["rewards"][index] for r in reward]
        traj["rewards"] = rewards[start_time : start_time + window]

        states = self.dataset["states"][index]
        states = [blosc.unpack_array(state)[:-1] for state in states[:-1]] + [
            blosc.unpack_array(states[-1])
        ]
        states = np.concatenate(states, axis=0)
        traj["states"] = states[start_time : start_time + window]

        timesteps = [len(actions) for actions in self.dataset["actions"][index]]
        gt_cls = [
            self.inst2cl[inst]
            for inst, t in zip(self.dataset["instructions"][index], timesteps)
            for _ in range(t)
        ]
        traj["gt_cls"] = gt_cls[start_time : start_time + window]

        return batch, video, traj


def label_unloc(config: dict) -> None:
    """
    Labels the trajectory segments based on the specified environment.

    Args:
        config (dict): Configuration parameters for labeling.

    Raises:
        NotImplementedError: If the specified environment is not implemented.
    """
    if config["env"] == "calvin":
        label_unloc_calvin(config)
    elif config["env"] == "babyai":
        label_unloc_babyai(config)
    else:
        raise NotImplementedError(
            f"This environment {config['env']} is not implemented"
        )


def label_unloc_babyai(config):
    """
    Label the extracted segments in the BabyAI dataset.

    Args:
        config (dict): Configuration parameters for the labeling process.

    Returns:
        None
    """
    # Index as run id
    index = random.randint(0, 1000)
    wandb.log({"Index": index})

    # Create visulisation_directory
    vis_directory = os.path.join(config["vis_directory"], "unloc", str(index))
    os.makedirs(vis_directory, exist_ok=True)

    # Load unsegmented data
    filename, ending = config["dataset_file"].split(".")
    clip_dataset = filename + "_clip." + ending
    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
        cl2inst_type = CL2INSTRUCTION_TYPE_GOTOLOCAL
    inst2cl = {value: key for key, value in cl2inst.items()}

    if not os.path.exists(os.path.join(config["dataset_directory"], clip_dataset)):
        dataset = transform2clip(config["dataset_file"], cl2inst, config)
        instruction_embeddings = dataset["instruction_embeddings"]
    else:
        with open(
            os.path.join(config["dataset_directory"], clip_dataset), "rb"
        ) as file:
            dataset = pickle.load(file)
            instruction_embeddings = dataset["instruction_embeddings"]

    # Load labelled data
    labelled_datafile = os.path.join(
        config["dataset_directory"], config["labelled_dataset_file"]
    )
    with open(labelled_datafile, "rb") as file:
        augmented_dataset = pickle.load(file)

    # Create Dataset
    dataset = LabellingDatasetUnLocBabyAI(
        dataset,
        config["max_window_size"],
        instruction_embeddings,
        inst2cl,
        config["device"],
    )

    # Load model
    model = UnLoc.load_from_checkpoint(
        config["checkpoint"], map_location=config["device"]
    )

    # Label segments
    n_labelled_segments = 0
    n_visualisations = 0
    accuracies = []
    segment_lengths = []
    instructions = []
    precision_start = []
    precision_end = []
    precision = []

    pbar = tqdm.tqdm(total=config["n_labelled_segments"])
    while n_labelled_segments < config["n_labelled_segments"]:
        batch, video, traj = dataset.sample_segment()

        probs = model.segment_segment(batch)
        new_samples, acc = postprocess_predictions_babyai(
            probs, traj, cl2inst, cl2inst_type
        )

        if new_samples == None:
            continue

        # Visualize labelled segment
        new_sample = random.choice(new_samples)
        if n_visualisations < config["n_visualisations"]:
            create_babyai_video(
                new_sample[1],
                new_sample[2],
                vis_directory,
                n_visualisations,
            )
            n_visualisations += 1

        for sample in new_samples:
            segment_lengths.append(sample[0].shape[0])
            instructions.append(sample[2])
            augmented_dataset["images"].append(blosc.pack_array(sample[0]))
            augmented_dataset["states"].append(blosc.pack_array(sample[1]))
            augmented_dataset["instructions"].append(sample[2])
            augmented_dataset["instruction_types"].append(sample[3])
            augmented_dataset["actions"].append(sample[4])
            augmented_dataset["directions"].append(sample[5])
            augmented_dataset["rewards"].append(sample[6])
            precision_start.append(sample[7])
            precision_end.append(sample[8])
            precision += [sample[7], sample[8]]

        pbar.update(len(new_samples))
        n_labelled_segments += len(new_samples)
        accuracies.append(acc)

    wandb.log({"Accuracy": sum(accuracies) / len(accuracies)})
    wandb.log({"Avg. Length": np.array(segment_lengths).mean()})
    wandb.log({"Precision": np.array(precision).mean()})
    wandb.log({"Precision Start": np.array(precision_start).mean()})
    wandb.log({"Precision End": np.array(precision_end).mean()})

    # Plot histogram of segment lengths
    segment_length_types = defaultdict(int)
    for segment_length in segment_lengths:
        segment_length_types[segment_length] += 1

    plt.bar(segment_length_types.keys(), segment_length_types.values())
    plt.xlabel("Segment Length")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"unloc_segment_length.png")
    plt.savefig(f"unloc_segment_length.pdf")

    # Plot histogram of instructions
    instruction_types = defaultdict(int)
    for instruction in instructions:
        instruction_types[instruction] += 1

    plt.clf()
    plt.bar(instruction_types.keys(), instruction_types.values())
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Instruction Types")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"unloc_instruction_types.png")
    plt.savefig(f"unloc_instruction_types.pdf")

    # Save augmented dataset
    filename = f"{config['labelled_dataset_file'].split('.')[0]}_unloc_{config['n_labelled_segments']}.pkl"
    unloc_dataset_file = os.path.join(config["dataset_directory"], filename)
    with open(unloc_dataset_file, "wb") as file:
        pickle.dump(augmented_dataset, file)


def postprocess_predictions_babyai(
    probs: torch.Tensor,
    traj: dict[str, Any],
    cl2inst: dict[int, str],
    cl2inst_type: dict,
) -> tuple[list, float]:
    """
    Postprocesses the predictions by extracting segmentation points, cutting out background segments,
    and computing pure subsegments.

    Args:
        probs (Tensor): The predicted probabilities for each class.
        traj (dict): The trajectory data containing observations, actions, directions, rewards, groundtruth classes, and states.
        cl2inst (dict): A mapping from class labels to instructions.
        cl2inst_type (dict): A mapping from class labels to instruction types.

    Returns:
        tuple: A tuple containing the list of samples and the overall accuracy.
            - samples (list): A list of samples, where each sample is a list containing the following elements:
                - obs (ndarray): The observations for the segment.
                - states (ndarray): The states for the segment.
                - instruction (str): The instruction for the segment.
                - instruction_type (dict): The instruction type for the segment.
                - actions (ndarray): The actions for the segment.
                - directions (ndarray): The directions for the segment.
                - rewards (ndarray): The rewards for the segment.
                - start_precision (bool): Indicates whether the start point is a segmentation point.
                - end_precision (bool): Indicates whether the end point is a segmentation point.
            - overall_acc (float): The overall accuracy of the pure subsegments.
    """
    traj_obs = traj["obs"]
    traj_actions = traj["actions"]
    traj_directions = traj["directions"]
    traj_rewards = traj["rewards"]
    traj_gt_cls = traj["gt_cls"]

    # From gt_cls extract segmentation points
    prev_cl = traj_gt_cls[0]
    starts = [0]
    ends = [traj_obs.shape[0]]
    for i, cl in enumerate(traj_gt_cls[1:]):
        if prev_cl != cl:
            starts.append(i + 1)
            ends.append(i)
            prev_cl = cl

    traj_states = traj["states"]
    predicted_cls = probs.argmax(dim=-1).squeeze(0)

    # Predict the class for each timestep and segment the trajectory
    predicted_cls = probs.argmax(dim=-1).squeeze(0)
    segmentation = cut_out_background(predicted_cls, cl2inst)

    # Compute all subsegments that are pure
    pure_segments = []
    for segment in segmentation:
        # Check whether the segment is pure
        start, end = segment
        pred_cls = predicted_cls[start:end].cpu().tolist()
        gt_cls = traj_gt_cls[start:end]
        acc = sum([p == g for p, g in zip(pred_cls, gt_cls)]) / len(gt_cls)
        if len(set(pred_cls)) > 1:
            continue

        cl = predicted_cls[start].cpu().item()
        pure_segments.append((start, end, cl, acc))

    # Make samples
    samples = []
    accs = []
    for pure_segment in pure_segments:
        start, end, cl, acc = pure_segment

        # Images
        obs = traj_obs[start:end].numpy()
        # Instructions
        instruction = cl2inst[cl]
        # Instruction Types
        instruction_type = cl2inst_type[cl].to_dict()
        # Actions
        actions = traj_actions[start:end]
        # Directions
        directions = traj_directions[start:end]
        # Rewards
        rewards = traj_rewards[start:end]
        # States
        states = traj_states[start:end]

        start_precision = start in starts
        end_precision = end in ends
        new_sample = [
            obs,
            states,
            instruction,
            instruction_type,
            actions,
            directions,
            rewards,
            start_precision,
            end_precision,
        ]

        samples.append(new_sample)
        accs.append(acc)

    if len(samples) > 0:
        overall_acc = sum(accs) / len(accs)
        return samples, overall_acc
    else:
        return None, None


def cut_out_background(
    predicted_cls: np.ndarray, cl2inst: dict[int, str]
) -> list[tuple[int, int]]:
    """
    Cuts out background segments from the predicted class labels.

    Args:
        predicted_cls (numpy.ndarray): Array of predicted class labels.
        cl2inst (dict): Dictionary mapping class labels to instances.

    Returns:
        list: List of segments representing the start and end indices of non-background segments.
    """
    bg_cls = len(cl2inst)
    segments = []
    start = -1
    end = -1

    for timestep in range(predicted_cls.shape[0] - 1):
        if predicted_cls[timestep] != bg_cls:
            end = timestep

        if predicted_cls[timestep] == bg_cls:
            if start < end:
                segments.append((start + 1, timestep))
                end = timestep
                start = end
            else:
                start = timestep
                end = timestep

    return segments


def label_unloc_calvin(config: dict) -> None:
    """
    Labels the trajectory segments of CALVIN using UnLoc.

    Args:
        config (dict): Configuration parameters for the labelling process.

    Returns:
        dict: The annotated trajectory segments with labels and other information.
    """
    # Index as run id
    index = random.randint(0, 1000)
    wandb.log({"Index": index})

    # Visualisation Directory
    visualisation_directory = os.path.join(config["vis_directory"], "unloc", str(index))
    os.makedirs(visualisation_directory, exist_ok=True)

    # Frame Directory
    frame_directory = os.path.join(config["dataset_directory"], "training")

    # Get annotation data
    annotation_directory = os.path.join(
        config["dataset_directory"], "training", "lang_annotations"
    )
    full_annotation = np.load(
        os.path.join(annotation_directory, "auto_lang_ann.npy"), allow_pickle=True
    ).item()
    partial_annotation = np.load(
        os.path.join(annotation_directory, f"auto_lang_ann_{config['percent']}.npy"),
        allow_pickle=True,
    ).item()
    gt_annotation = {
        "info": {"indx": []},
        "language": {
            "task": [],
            "ann": [],
            "emb": [],
            "task_emb": [],
            "task_frame2row": [],
            "clip_frame2row": [],
            "clip": [],
        },
    }
    clip_instructions = torch.from_numpy(full_annotation["language"]["clip"])
    clip_instructions = torch.cat(
        [clip_instructions, torch.zeros(1, clip_instructions.shape[1])], dim=0
    )

    # Create row2task, instruction2emb, task2instructions
    frame2task = {}
    for frame, task in zip(
        full_annotation["info"]["indx"], full_annotation["language"]["task"]
    ):
        frame2task[frame] = task

    row2task = {}
    frame2row = full_annotation["language"]["clip_frames2row"]
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

    # Get episodes
    frame_directory = os.path.join(config["dataset_directory"], "training")
    episodes = np.load(os.path.join(frame_directory, "ep_start_end_ids.npy"))
    n_episodes = episodes.shape[0]

    # Get UnLoc Model
    unloc_model = UnLoc.load_from_checkpoint(
        config["checkpoint"], map_location=config["device"]
    )
    unloc_model.eval()

    if config["debug"]:
        indeces = full_annotation["info"]["indx"][:10]
        annotations = full_annotation["language"]["ann"][:10]
        tasks = full_annotation["language"]["task"][:10]
        embds = full_annotation["language"]["emb"][:10]
    else:
        indeces = full_annotation["info"]["indx"]
        annotations = full_annotation["language"]["ann"]
        tasks = full_annotation["language"]["task"]
        embds = full_annotation["language"]["emb"]

    statistics = {"total": 0, "accepted": 0, "probs": []}
    pbar = tqdm.tqdm(total=len(full_annotation["language"]["ann"]))
    n_visualisations = 0
    for frame, ann, task, emb in zip(indeces, annotations, tasks, embds):
        done = False
        if frame in partial_annotation["info"]["indx"]:
            # Copy the stuff
            gt_annotation["info"]["indx"].append(frame)
            gt_annotation["language"]["ann"].append(ann)
            gt_annotation["language"]["emb"].append(emb)
            gt_annotation["language"]["task"].append(task)
            pbar.update(1)
        else:
            while not done:
                # Sample episode
                eps = random.randint(0, n_episodes - 1)
                start, end = episodes[eps]

                # Sample start point
                start = random.randint(start, end - config["window_size"] - 1)

                # Load data
                sample = load_data(
                    (start, start + config["window_size"]), frame_directory
                )
                if not sample:
                    continue

                # Crop segment
                sample["instructions"] = clip_instructions
                log_probs = unloc_model.get_log_probs(sample, config["device"])
                label_prediction = log_probs.argmax(dim=-1)
                label_prediction = label_prediction.squeeze(0)

                # Process Prediction
                postprocessed_segment = postprocess_predictions_calvin(
                    label_prediction, start, config["window_size"]
                )
                if postprocessed_segment == None:
                    continue
                frame, predicted_row = postprocessed_segment

                task = row2task[predicted_row]
                ann = random.sample(task2instructions[task], k=1)[0]
                emb = instruction2emb[ann]

                gt_annotation["info"]["indx"].append(frame)
                gt_annotation["language"]["ann"].append(ann)
                gt_annotation["language"]["emb"].append(emb)
                gt_annotation["language"]["task"].append(task)

                # Update statistics
                statistics["accepted"] += 1
                pbar.update(1)
                done = True

                # Update statistics
                statistics["total"]

                # Visualize Segment
                if n_visualisations < config["n_visualisations"]:
                    video = sample["rgb_obs"]
                    video_path = os.path.join(
                        visualisation_directory, f"{n_visualisations}_{ann}.mp4"
                    )
                    write_video(video_path, video, fps=15)
                    n_visualisations += 1

    # Postprocess instruction embeddings
    gt_annotation["language"]["emb"] = np.concatenate(
        gt_annotation["language"]["emb"], axis=0
    )

    # Output statistics
    if statistics["total"] > 0:
        print(
            f"Percentage of accepted segments {statistics['accepted']/statistics['total']}"
        )
    if len(statistics["probs"]) > 0:
        print(
            f"Average probability {sum(statistics['probs'])/len(statistics['probs'])} "
        )

    # Save statistics
    np.save(
        os.path.join(
            annotation_directory,
            f"auto_lang_ann_unloc_{config['threshold']}_{config['percent']}_statistics.pkl",
        ),
        statistics,
        allow_pickle=True,
    )

    # Save random annototation
    np.save(
        os.path.join(
            annotation_directory,
            f"auto_lang_ann_unloc_{config['threshold']}_{config['percent']}.npy",
        ),
        gt_annotation,
        allow_pickle=True,
    )


def postprocess_predictions_calvin(
    label_prediction: torch.Tensor, start: int, window_size: int
) -> tuple | None:
    """
    Postprocesses the label predictions to extract valid segments.

    Args:
        label_prediction (torch.Tensor): The predicted labels.
        start (int): The start timestep of the segment.
        window_size (int): The size of the sliding window.

    Returns:
        tuple | None: A tuple containing the frame range and the most frequent label in the segment,
                      or None if the segment is not valid.
    """
    bg_mask = label_prediction == 34
    end = start + window_size
    start_found = False
    end_found = False
    start_segment = None
    end_segment = None

    # Cut off any background segments from the left and right
    for i in range(window_size):
        if not start_found:
            if not bg_mask[i]:
                start_segment = i
                start_found = True

        if not end_found:
            if not bg_mask[window_size - 1 - i]:
                end_segment = i
                end_found = True

    # Check whether the segment is valid
    if not start_segment and not end_segment:
        return None
    if start + start_segment >= end - end_segment:
        return None

    # Get frame range
    end = start + window_size - end_segment
    start = start + start_segment
    frame = (start, end)

    # Most frequent label in segment
    label2freq = defaultdict(int)
    for label in label_prediction:
        label2freq[label.item()] += 1
    predicted_row = [
        item[0] for item in sorted(label2freq.items(), key=lambda item: item[1])
    ][-1]

    # Skip if background segment
    if predicted_row == 34:
        return None

    # Valid segment checks

    ## Consists only of one label
    if (
        len(label_prediction[start_segment : start_segment + (end - start)].unique())
        != 1
    ):
        logging.info("Not just one label")
        return None

    ## Does not have the minimum length
    if end - start < config["min_window_size"]:
        logging.info(f"No minimum length {end-start}")
        return None

    return frame, predicted_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/labelling_unloc.yaml",
        help="Location of config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Location of checkpoint file",
    )
    args = parser.parse_args()

    # Config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Update with CLI arguments
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint

    # Wandb Login
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"],
        dir="../results",
    )

    label_unloc(config)
