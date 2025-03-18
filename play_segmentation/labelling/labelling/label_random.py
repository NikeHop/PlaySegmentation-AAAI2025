import argparse
import os
import pickle
import random

from collections import defaultdict

import blosc
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
import yaml

from torchvision.io import write_video
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from play_segmentation.labelling.labelling.utils import load_data, preprocess_sample
from play_segmentation.labelling.labelling_training.trainer import I3D
from play_segmentation.data.babyai.utils import (
    CL2INSTRUCTION_GOTOLOCAL,
    CL2INSTRUCTION_TYPE_GOTOLOCAL,
)
from play_segmentation.labelling.labelling.utils import create_babyai_video


class LabellingRandomBabyAI(Dataset):
    """
    Dataset class for random labelling in BabyAI environment.

    Args:
        dataset (dict): The dataset containing instructions, images, actions, rewards, directions, and states.
        max_window (int): The maximum window size for sampling random segments.
        device (str): The device to be used for computation.

    Attributes:
        dataset (dict): The dataset containing instructions, images, actions, rewards, directions, and states.
        max_window (int): The maximum window size for sampling random segments.
        length (int): The length of the dataset.
        device (str): The device to be used for computation.
    """

    def __init__(self, dataset: dict, max_window: int, device: str) -> None:
        self.dataset = dataset
        self.max_window = max_window
        self.length = len(self.dataset["instructions"])
        self.device = device

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves a random segment from the dataset.

        Args:
            index (int): The index of the segment to retrieve.

        Returns:
            tuple: A tuple containing the video, states, actions, rewards, and directions of the segment.
        """
        segments = [
            torch.tensor(blosc.unpack_array(segment), dtype=torch.float)
            for segment in self.dataset["images"][index]
        ]
        traj = torch.cat(segments, dim=0)

        T = traj.shape[0]

        # Sample a random window
        random_window = random.randint(2, min(self.max_window, T - 2))
        start = random.randint(0, T - random_window - 2)
        end = start + random_window

        # Prepare data for labelling model
        video = traj[start:end]
        actions = [a for action in self.dataset["actions"][index] for a in action]
        actions = actions[start:end]
        rewards = [r for reward in self.dataset["rewards"][index] for r in reward]
        rewards = rewards[start:end]
        directions = [
            d for directions in self.dataset["directions"][index] for d in directions
        ]
        directions = directions[start:end]
        states = np.concatenate(
            [blosc.unpack_array(segment) for segment in self.dataset["states"][index]],
            axis=0,
        )
        states = states[start:end]

        return video, states, actions, rewards, directions

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.length


def label_random(config: dict) -> None:
    """
    Randomly labels the given configuration based on the environment.

    Parameters:
        config (dict): The configuration dictionary containing the environment information.

    Raises:
        NotImplementedError: If the environment specified in the configuration is not implemented.

    """
    if config["env"] == "babyai":
        label_random_babyai(config)
    elif config["env"] == "calvin":
        label_random_calvin(config)
    else:
        raise NotImplementedError(f"Environment {config['env']} not implemented")


def label_random_calvin(config: dict) -> None:
    """
    Randomly labels segments of data based on a given configuration.

    Args:
        config (dict): Configuration parameters for the labeling process.

    Returns:
        None
    """
    # Sample run index
    index = random.randint(0, 100000)
    wandb.log({"Index": index})

    # Visualisation Directory
    visualisation_directory = os.path.join(
        config["visualisation_directory"], "random", str(index)
    )
    os.makedirs(visualisation_directory, exist_ok=True)

    # Load data
    annotation_directory = os.path.join(
        config["dataset_directory"], "training", "lang_annotations"
    )

    full_annotation = np.load(
        os.path.join(annotation_directory, "auto_lang_ann.npy"), allow_pickle=True
    ).item()

    # Get annotated data
    partial_annotation = np.load(
        os.path.join(annotation_directory, f"auto_lang_ann_{config['percent']}.npy"),
        allow_pickle=True,
    ).item()
    partial_annotation["language"]["emb"] = list(partial_annotation["language"]["emb"])

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

    # Get episode data
    frame_directory = os.path.join(config["dataset_directory"], "training")
    episodes = np.load(os.path.join(frame_directory, "ep_start_end_ids.npy"))
    n_episodes = episodes.shape[0]

    # Get Labelling Model
    model = I3D.load_from_checkpoint(
        config["i3d_checkpoint"], map_location=config["device"]
    )
    model.eval()

    # Annotate data
    statistics = {"total": 0, "accepted": 0, "probs": []}
    if config["debug"]:
        n_added_samples = 10
    else:
        n_added_samples = len(full_annotation["language"]["ann"]) - len(
            partial_annotation["language"]["ann"]
        )
    pbar = tqdm.tqdm(total=n_added_samples)
    n_visualisations = 0
    while statistics["accepted"] < n_added_samples:
        # Sample episode
        eps = random.randint(0, n_episodes - 1)
        start, end = episodes[eps]

        # Sample start point
        start = random.randint(start, end - config["max_window_size"] - 1)

        # Sample window_size
        ws = random.randint(config["min_window_size"], config["max_window_size"])

        # Load data
        sample = load_data((start, start + ws), frame_directory)
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
            frame = (start, start + ws)
            task = row2task[predicted_row.item()]
            ann = random.sample(task2instructions[task], k=1)[0]
            emb = instruction2emb[ann].squeeze(0)

            partial_annotation["info"]["indx"].append(frame)
            partial_annotation["language"]["ann"].append(ann)
            partial_annotation["language"]["emb"].append(emb)
            partial_annotation["language"]["task"].append(task)

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
    partial_annotation["language"]["emb"] = np.stack(
        partial_annotation["language"]["emb"], axis=0
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
            f"auto_lang_ann_random_{config['threshold']}_{config['percent']}_statistics.pkl",
        ),
        statistics,
        allow_pickle=True,
    )

    # Save random annototation
    np.save(
        os.path.join(
            annotation_directory,
            f"auto_lang_ann_random_{config['threshold']}_{config['percent']}_{config['labelling_model']}.npy",
        ),
        partial_annotation,
        allow_pickle=True,
    )


def label_random_babyai(config: dict) -> None:
    """
    Randomly labels segments in a BabyAI dataset.

    Args:
        config (dict): Configuration parameters for the labeling process.

    Returns:
        None
    """
    # Random index of the run
    index = random.randint(0, 100000)
    wandb.log({"Index": index})

    # Visualisation
    visualisation_directory = os.path.join(
        config["visualisation_directory"], "random", str(index)
    )
    os.makedirs(visualisation_directory, exist_ok=True)

    # Unsegmented Dataset
    with open(
        os.path.join(config["dataset_directory"], config["dataset_file"]), "rb"
    ) as file:
        dataset = pickle.load(file)

    dataset = LabellingRandomBabyAI(dataset, config["max_window"], config["device"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=10,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    # Segmented Dataset
    with open(
        os.path.join(config["dataset_directory"], config["labelled_dataset_file"]), "rb"
    ) as file:
        augmented_dataset = pickle.load(file)

    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
        cl2inst_type = CL2INSTRUCTION_TYPE_GOTOLOCAL

    model = I3D.load_from_checkpoint(
        config["i3d_checkpoint"], map_location=config["device"]
    )
    model.eval()

    n_labelled_segments = 0
    n_visualisations = 0
    pbar = tqdm.tqdm(total=config["n_labelled_segments"])
    while n_labelled_segments < config["n_labelled_segments"]:
        for data in dataloader:
            actions_list = []
            rewards_list = []
            directions_list = []
            states_list = []
            videos = []

            for sample in data:
                video, states, actions, rewards, directions = sample
                actions_list.append(actions)
                rewards_list.append(rewards)
                directions_list.append(directions)
                states_list.append(states)
                videos.append(video.to(config["device"]))

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
            for (
                video,
                state,
                instruction,
                inst_type,
                actions,
                rewards,
                directions,
            ) in zip(
                videos,
                states_list,
                instructions,
                inst_types,
                actions_list,
                rewards_list,
                directions_list,
            ):
                augmented_dataset["images"].append(
                    blosc.pack_array(video.cpu().numpy())
                )
                augmented_dataset["states"].append(blosc.pack_array(state))
                augmented_dataset["instructions"].append(instruction)
                augmented_dataset["instruction_types"].append(inst_type.to_dict())
                augmented_dataset["actions"].append(actions)
                augmented_dataset["directions"].append(directions)
                augmented_dataset["rewards"].append(rewards)

            n_labelled_segments += config["batch_size"]
            if n_labelled_segments >= config["n_labelled_segments"]:
                break
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
    filename = f"{config['labelled_dataset_file'].split('.')[0]}_random_{config['n_labelled_segments']}_{config['labelling_model']}.pkl"
    unloc_dataset_file = os.path.join(config["dataset_directory"], filename)
    with open(unloc_dataset_file, "wb") as file:
        pickle.dump(augmented_dataset, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to experiment config file", type=str)
    parser.add_argument("--i3d-checkpoint", help="path to i3d checkpoint", type=str)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Add CLI arguments
    if args.i3d_checkpoint:
        config["i3d_checkpoint"] = args.i3d_checkpoint

    # Prepare Logging
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"],
        dir="../results",
    )
    label_random(config)
