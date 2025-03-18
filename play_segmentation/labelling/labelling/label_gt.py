import argparse
import os
import pickle
import random

from collections import defaultdict

import blosc
import numpy as np
import torch
import tqdm
import wandb
import yaml

from torchvision.io import write_video
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


from play_segmentation.data.babyai.utils import (
    CL2INSTRUCTION_GOTOLOCAL,
    CL2INSTRUCTION_TYPE_GOTOLOCAL,
)

from play_segmentation.labelling.labelling.utils import load_data, preprocess_sample
from play_segmentation.labelling.labelling_training.trainer import I3D
from play_segmentation.labelling.labelling.utils import create_babyai_video


class LabellingGT(Dataset):
    """
    Dataset class for ground truth labelling.

    Args:
        dataset (dict): The dataset containing the instructions, images, actions, rewards, directions, and states.
        device (torch.device): The device to be used for tensor operations.

    Attributes:
        dataset (dict): The dataset containing the instructions, images, actions, rewards, directions, and states.
        length (int): The length of the dataset.
        device (torch.device): The device to be used for tensor operations.
    """

    def __init__(self, dataset: dict, device: torch.device) -> None:
        self.dataset = dataset
        self.length = len(self.dataset["instructions"])
        self.device = device

    def sample_segment(self) -> tuple:
        """
        Samples a segment from the dataset.

        Returns:
            tuple: A tuple containing the video, states, actions, rewards, and directions of the sampled segment.
        """
        index = random.randint(0, self.length - 1)
        n_segments = self.dataset["images"][index]
        segment_index = random.randint(0, len(n_segments) - 1)

        # Prepare data for labelling model
        video = torch.tensor(
            blosc.unpack_array(self.dataset["images"][index][segment_index]),
            dtype=torch.float,
        )
        actions = [a for a in self.dataset["actions"][index][segment_index]]
        rewards = [r for r in self.dataset["rewards"][index][segment_index]]
        directions = [d for d in self.dataset["directions"][index][segment_index]]
        states = blosc.unpack_array(self.dataset["states"][index][segment_index])

        return video.to(self.device), states, actions, rewards, directions

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.length)


def label_gt(config: dict) -> None:
    """
    Relabels ground truth data based on the given configuration.

    Args:
        config (dict): Configuration dictionary containing the environment information.

    Raises:
        NotImplementedError: If the specified environment is not implemented.
    """
    if config["env"] == "babyai":
        label_gt_babyai(config)
    elif config["env"] == "calvin":
        label_gt_calvin(config)
    else:
        raise NotImplementedError(f"Environment {config['env']} not implemented")


def label_gt_babyai(config: dict) -> None:
    """
    Labels the ground truth (gt) dataset using the BabyAI environment.

    Args:
        config (dict): Configuration parameters for labeling the dataset.

    Raises:
        NotImplementedError: If the specified environment is not implemented.

    Returns:
        None
    """
    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
        cl2inst_type = CL2INSTRUCTION_TYPE_GOTOLOCAL
    else:
        raise NotImplementedError(f"Environment {config['env']} not implemented")

    # Random index to distinguish between runs
    index = random.randint(0, 100000)
    wandb.log({"Index": index})

    # Visualisation
    visualisation_directory = os.path.join(
        config["visualisation_directory"], "gt", str(index)
    )
    if not os.path.exists(visualisation_directory):
        os.makedirs(visualisation_directory)

    # Unsegmented Dataset
    with open(
        os.path.join(config["dataset_directory"], config["dataset_file"]), "rb"
    ) as file:
        dataset = pickle.load(file)
    dataset = LabellingGT(dataset, config["device"])

    # Segmented Dataset
    with open(
        os.path.join(config["dataset_directory"], config["labelled_dataset_file"]), "rb"
    ) as file:
        augmented_dataset = pickle.load(file)

    # Load model
    model = I3D.load_from_checkpoint(
        config["i3d_checkpoint"], map_location=config["device"]
    )
    model.eval()

    max_n_labelled_segments = len(augmented_dataset["instructions"]) * (
        config["augmentation_factor"] - 1
    )
    n_labelled_segments = 0
    n_visualisations = 0
    pbar = tqdm.tqdm(total=max_n_labelled_segments)
    while n_labelled_segments < max_n_labelled_segments:
        actions_list = []
        rewards_list = []
        directions_list = []
        states_list = []
        videos = []
        for i in range(config["batch_size"]):
            video, states, actions, rewards, directions = dataset.sample_segment()
            actions_list.append(actions)
            rewards_list.append(rewards)
            directions_list.append(directions)
            states_list.append(states)
            videos.append(video)

        videos = pad_sequence(videos, batch_first=True)

        with torch.no_grad():
            cl = model.label(videos)

        cl = cl.cpu()

        instructions = []
        inst_types = []
        for c in cl:
            c = c.item()
            instruction = cl2inst[c]
            inst_type = cl2inst_type[c]
            instructions.append(instruction)
            inst_types.append(inst_type)

        # Create new sample
        for video, states, instruction, inst_type, actions, rewards, directions in zip(
            videos,
            states_list,
            instructions,
            inst_types,
            actions_list,
            rewards_list,
            directions_list,
        ):
            augmented_dataset["images"].append(blosc.pack_array(video.cpu().numpy()))
            augmented_dataset["states"].append(blosc.pack_array(states))
            augmented_dataset["instructions"].append(instruction)
            augmented_dataset["instruction_types"].append(inst_type.to_dict())
            augmented_dataset["actions"].append(actions)
            augmented_dataset["directions"].append(directions)
            augmented_dataset["rewards"].append(rewards)

        n_labelled_segments += config["batch_size"]
        pbar.update(config["batch_size"])

        if n_visualisations < config["n_visualisations"]:
            create_babyai_video(
                states_list[0],
                instructions[0],
                visualisation_directory,
                n_visualisations,
            )
            n_visualisations += 1

    # Save augmented dataset
    filename = (
        f"{config['labelled_dataset_file'].split('.')[0]}_gt_{n_labelled_segments}.pkl"
    )
    unloc_dataset_file = os.path.join(config["dataset_directory"], filename)
    with open(unloc_dataset_file, "wb") as file:
        pickle.dump(augmented_dataset, file)


def label_gt_calvin(config: dict) -> None:
    """
    Labels ground truth data based on a given configuration.

    Args:
        config (dict): Configuration parameters for labeling.

    Returns:
        None
    """
    # Visualisation Directory
    visualisation_directory = os.path.join(config["visualisation_directory"], "gt")
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
            "task_frames2row": [],
            "clip_frames2row": [],
            "clip": [],
        },
    }

    # Create row2task, instruction2emb, task2instructions
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

    # Get Labelling Model
    model = I3D.load_from_checkpoint(
        config["i3d_checkpoint"], map_location=config["device"]
    )
    model.eval()

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
        if frame in partial_annotation["info"]["indx"]:
            # Copy the stuff
            gt_annotation["info"]["indx"].append(frame)
            gt_annotation["language"]["ann"].append(ann)
            gt_annotation["language"]["emb"].append(emb)
            gt_annotation["language"]["task"].append(task)
            pbar.update(1)
        else:
            start, end = frame

            # Load data
            sample = load_data((start, end), frame_directory)
            if not sample:
                continue
            batch = preprocess_sample(sample, config)

            # Label segment
            probs = model.model(batch)
            predicted_row = probs.argmax(dim=-1)
            predicted_prob, _ = probs.max(dim=-1)

            # Collect data
            statistics["total"] += 1
            statistics["probs"].append(predicted_prob.item())

            if predicted_prob > config["threshold"]:
                frame = (start, end)
                task = row2task[predicted_row.item()]
                ann = random.sample(task2instructions[task], k=1)[0]
                emb = instruction2emb[ann]

                gt_annotation["info"]["indx"].append(frame)
                gt_annotation["language"]["ann"].append(ann)
                gt_annotation["language"]["emb"].append(emb)
                gt_annotation["language"]["task"].append(task)

                # Update statistics
                statistics["accepted"] += 1
                pbar.update(1)

                # Visualise examples
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
            f"auto_lang_ann_gt_{config['threshold']}_{config['percent']}_statistics.pkl",
        ),
        statistics,
        allow_pickle=True,
    )

    # Save random annotations
    np.save(
        os.path.join(
            annotation_directory,
            f"auto_lang_ann_gt_{config['threshold']}_{config['percent']}.npy",
        ),
        gt_annotation,
        allow_pickle=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to experiment config file", type=str)
    parser.add_argument("--i3d-checkpoint", help="path to i3d checkpoint", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if args.i3d_checkpoint:
        config["i3d_checkpoint"] = args.i3d_checkpoint

    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"],
    )

    label_gt(config)
