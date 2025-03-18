import os
import pickle
import random

import blosc
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from play_segmentation.data.babyai.utils import (
    CL2INSTRUCTION_GOTOLOCAL,
)
from play_segmentation.utils.data import BaseDataset


def get_data_play_segmentation(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Get data for play segmentation based on the given configuration.

    Args:
        config (dict): Configuration parameters for the data.

    Returns:
        data: Data for play segmentation.

    Raises:
        NotImplementedError: If the environment is not supported.
    """
    if config["env"] == "babyai":
        return get_data_play_segmentation_babyai(config)
    elif config["env"] == "calvin":
        return get_data_play_segmentation_calvin(config)
    else:
        raise NotImplementedError("This environment is not supported")


def get_data_play_segmentation_babyai(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Get the data loaders for training and validation datasets for play segmentation in BabyAI environment.

    Args:
        config (dict): Configuration parameters for the data loading.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and validation data loaders.
    """

    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
    else:
        raise NotImplementedError(f"Environment {config['env_name']} is not supported")

    training_dataset_file = os.path.join(
        config["dataset_directory"], config["filename"] + "_training.pkl"
    )
    validation_dataset_file = os.path.join(
        config["dataset_directory"], config["filename"] + "_validation.pkl"
    )

    with open(training_dataset_file, "rb") as file:
        training_dataset = pickle.load(file)

    with open(validation_dataset_file, "rb") as file:
        validation_dataset = pickle.load(file)

    inst2cl = {v: k for k, v in cl2inst.items()}
    training_dataset = PlaySegmentationBabyAI(
        training_dataset, config["buffer"], config["max_distance"], inst2cl
    )
    validation_dataset = PlaySegmentationBabyAI(
        validation_dataset, config["buffer"], config["max_distance"], inst2cl
    )

    train_dataloader = DataLoader(
        dataset=training_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=collate_augmented_segments,
        num_workers=config["num_workers"],
    )
    val_dataloader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=collate_augmented_segments,
        num_workers=config["num_workers"],
    )

    return train_dataloader, val_dataloader


def get_data_play_segmentation_calvin(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Retrieves and prepares the training and validation data for Play Segmentation in the CALVIN environment.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and validation data loaders.
    """
    frames2split = {}
    # Training Episodes
    training_directory = os.path.join(config["dataset_directory"], "training")
    with open(os.path.join(training_directory, "ep_start_end_ids.npy"), "rb") as file:
        training_episodes = np.load(file)
    for eps in training_episodes:
        for frame in range(eps[0], eps[1] + 1):
            frame = "0" * (7 - len(str(frame))) + str(frame)
            frames2split[frame] = "training"

    # Validation episodes
    validation_directory = os.path.join(config["dataset_directory"], "validation")
    with open(os.path.join(validation_directory, "ep_start_end_ids.npy"), "rb") as file:
        validation_episodes = np.load(file)
    for eps in validation_episodes:
        for frame in range(eps[0], eps[1] + 1):
            frame = "0" * (7 - len(str(frame))) + str(frame)
            frames2split[frame] = "validation"

    episodes = np.concatenate([training_episodes, validation_episodes], axis=0)
    split2episodes = {
        "training": training_episodes,
        "validation": validation_episodes,
    }

    datasets = {}
    for split in ["training", "validation"]:
        episode_directory = os.path.join(config["dataset_directory"], split)

        if split == "training":
            filename = f"{config['episode_filename']}"
        else:
            filename = "auto_lang_ann.npy"

        with open(
            os.path.join(episode_directory, f"lang_annotations/{filename}"),
            "rb",
        ) as file:
            lang = np.load(file, allow_pickle=True).item()
            frames2row = {
                frame: lang["language"]["task_frames2row"][frame]
                for frame in lang["info"]["indx"]
            }

        dataset = PlaySegmentationCalvin(
            split,
            config["dataset_directory"],
            episodes,
            split2episodes[split],
            frames2row,
            frames2split,
            config["buffer"],
            config["max_distance"],
        )

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=config["batch_size"],
            collate_fn=collate_augmented_segments,
            num_workers=config["num_workers"],
        )

        datasets[split] = dataloader

    return datasets["training"], datasets["validation"]


def collate_augmented_segments(data: list) -> dict:
    """
    Collates augmented segments data into a batch.

    Args:
        data (list): List of samples, where each sample is a tuple of three elements:
                     - List of observation sequences
                     - List of labels
                     - List of class labels

    Returns:
        dict: A dictionary containing the collated data with the following keys:
              - "img_obs": Tensor of padded observation sequences
              - "obs_length": Tensor of lengths of each observation sequence
              - "labels": Tensor of labels
              - "class_labels": Tensor of class labels
    """
    obss = []
    obs_length = []
    labels = []
    rows = []

    for sample in data:
        obss += [torch.tensor(elem).float() for elem in sample[0]]
        obs_length += [elem.shape[0] for elem in sample[0]]
        labels += sample[1]
        rows += sample[2]

    obss = pad_sequence(obss, batch_first=True)
    obs_length = torch.tensor(obs_length, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    rows = torch.tensor(rows, dtype=torch.long)

    return {
        "img_obs": obss,
        "obs_length": obs_length,
        "labels": labels,
        "class_labels": rows,
    }


class PlaySegmentationBabyAI(Dataset):
    """
    Dataset class for play segmentation in BabyAI.

    Args:
        dataset (dict): The dataset containing images and instructions.
        buffer (int): The buffer size for augmentation.
        max_distance (int): The maximum distance for augmentation.
        inst2cl (dict): A mapping from instructions to class labels.
    """

    def __init__(
        self, dataset: dict, buffer: int, max_distance: int, inst2cl: dict
    ) -> None:
        self.dataset = dataset
        self.buffer = buffer
        self.max_distance = max_distance
        self.inst2cl = inst2cl
        self.length = len(self.dataset["images"])

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.length

    def __getitem__(self, index: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the segmented observations, labels, and instructions.
        """
        # Sample an unsegmented trajectory
        unseg_traj = self.dataset["images"][index]
        instructions = self.dataset["instructions"][index]

        # Unpack trajectory and transform it into array
        segments = [blosc.unpack_array(elem) for elem in unseg_traj]
        obs_seq = np.concatenate(segments, axis=0)

        # Determine length of unsegmented trajectory
        length = obs_seq.shape[0]
        timesteps = np.array([0] + [segment.shape[0] for segment in segments]).cumsum()

        # Sample a segment from the unsegmented trajectory
        index = random.randint(0, len(unseg_traj) - 1)
        start_time = timesteps[index]
        end_time = timesteps[index + 1]
        instruction = instructions[index]
        cl = self.inst2cl[instruction]

        labels = [1]
        instructions = [cl]
        segments = [(start_time, end_time)]

        # Augmentation to the right
        if random.random() > 0.5:
            # Later end-time
            max_add_on = min(self.max_distance, length - end_time)
            if self.buffer < max_add_on:
                translation_steps_right = random.randint(self.buffer, max_add_on)
                segments.append((start_time, end_time + translation_steps_right))
                labels.append(0)
                instructions.append(-1)
        else:
            max_subtract = min(self.max_distance, end_time - start_time - 1)
            if self.buffer < max_subtract:
                # Earlier end-time
                translation_steps_right = random.randint(self.buffer, max_subtract)
                segments.append((start_time, end_time - translation_steps_right))
                labels.append(0)
                instructions.append(-1)

        # Augmentation both sides
        max_subtract = min(self.max_distance, start_time)
        max_add_on = min(self.max_distance, length - end_time)

        if self.buffer < max_subtract and self.buffer < max_add_on:
            translation_steps_left = random.randint(self.buffer, max_subtract)
            translation_steps_right = random.randint(self.buffer, max_add_on)
            segments.append(
                (
                    start_time - translation_steps_left,
                    end_time + translation_steps_right,
                )
            )
            labels.append(0)
            instructions.append(-1)

        # Augmentation to the left
        max_subtract = min(start_time, self.max_distance)
        if self.buffer < max_subtract:
            translation_steps_left = random.randint(self.buffer, max_subtract)
            segments.append((start_time - translation_steps_left, start_time))
            labels.append(0)
            instructions.append(-1)

        # Translate to the left
        max_translation_steps = min(self.max_distance, start_time)
        if self.buffer < max_translation_steps:
            # Check whether it is possible to translate
            translation_steps_left = random.randint(self.buffer, max_translation_steps)
            segments.append(
                (start_time - translation_steps_left, end_time - translation_steps_left)
            )
            labels.append(0)
            instructions.append(-1)

        # Translate to the right
        max_translation_steps = min(self.max_distance, length - end_time)
        if self.buffer < max_translation_steps:
            # Check whether it is possible to translate
            translation_steps_right = random.randint(self.buffer, max_translation_steps)
            segments.append(
                (
                    start_time + translation_steps_right,
                    end_time + translation_steps_right,
                )
            )
            labels.append(0)
            instructions.append(-1)

        obss = []
        for segment in segments:
            start = segment[0]
            end = segment[1]
            obs = obs_seq[start:end]
            obss.append(obs)

        return obss, labels, instructions


class PlaySegmentationCalvin(BaseDataset):
    """
    Dataset class for Play Segmentation using the Calvin dataset.

    Args:
        split (str): The split of the dataset.
        dataset_directory (str): The directory where the dataset is stored.
        episodes (list): List of episodes, each represented as a tuple of start and end frames.
        split_episodes (dict): Dictionary mapping frames to their corresponding split.
        frames2row (dict): Dictionary mapping frames to their corresponding instruction class.
        frames2split (dict): Dictionary mapping frames to their corresponding split.
        buffer (int): The buffer size for displacement sampling.
        max_distance (int): The maximum distance for displacement sampling.

    Attributes:
        data_keys (list): List of keys for the data.
        buffer (int): The buffer size for displacement sampling.
        max_distance (int): The maximum distance for displacement sampling.

    Methods:
        __len__: Returns the length of the dataset.
        __getitem__: Gets an item from the dataset.
    """

    def __init__(
        self,
        split: str,
        dataset_directory: str,
        episodes: list,
        split_episodes: dict,
        frames2row: dict,
        frames2split: dict,
        buffer: int,
        max_distance: int,
    ) -> None:
        self.data_keys: list[str] = ["rgb_static"]
        self.buffer: int = buffer
        self.max_distance: int = max_distance

        super().__init__(
            split,
            dataset_directory,
            episodes,
            split_episodes,
            frames2row,
            frames2split,
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of elements in the dataset.
        """
        return len(self.frames2row)

    def __getitem__(self, index: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            obss (list): List of observations for different subsegments.
            labels (list): List of labels indicating if the subsegment is longer or shorter.
            rows (list): List of rows containing instructions classes for the segments.
        """

        # Get a segment
        start_frame, end_frame = random.choice(list(self.frames2row.keys()))

        # Get instructions
        row = self.frames2row[(start_frame, end_frame)]
        rows = [row]

        # Get the corresponding episode for the segment
        eps_start_frame, eps_end_frame = self.frames2episode[(start_frame, end_frame)]

        # Sample an extended timeframe
        max_time_right = min(eps_end_frame - end_frame, self.max_distance)
        max_time_left = min(start_frame - eps_start_frame, self.max_distance)

        # Sample subtimeframes longer and shorter
        trajs = [(start_frame, end_frame)]
        labels = [end_frame - start_frame]

        # Sample Right Boundary
        if max_time_right > self.buffer:
            add_on_right = random.randint(self.buffer + 1, max_time_right)
            if random.random() > 0.5:
                aug_end_frame = end_frame + add_on_right
            else:
                aug_end_frame = end_frame - add_on_right
        else:
            aug_end_frame = end_frame

        # Sample Left Boundary
        if max_time_left > self.buffer:
            add_on_left = random.randint(self.buffer + 1, max_time_left)
            if random.random() > 0.5:
                aug_start_frame = start_frame + add_on_left
            else:
                aug_start_frame = start_frame - add_on_left
        else:
            aug_start_frame = start_frame

        if start_frame != aug_start_frame:
            trajs.append((aug_start_frame, end_frame))
            labels.append(0)

        if end_frame != aug_end_frame:
            trajs.append((start_frame, aug_end_frame))
            labels.append(0)

        if (
            aug_start_frame != start_frame or aug_end_frame != end_frame
        ) and aug_start_frame < aug_end_frame:
            trajs.append((aug_start_frame, aug_end_frame))
            labels.append(0)

        timesteps_to_translate_to_right = random.randint(self.buffer, self.max_distance)
        timesteps_to_translate_to_left = random.randint(self.buffer, self.max_distance)

        if start_frame - timesteps_to_translate_to_left >= eps_start_frame:
            # Translate to the right
            trajs.append(
                (
                    start_frame - timesteps_to_translate_to_left,
                    end_frame - timesteps_to_translate_to_left,
                )
            )
            labels.append(0)

        if end_frame + timesteps_to_translate_to_right <= eps_end_frame:
            # Translate to the left
            trajs.append(
                (
                    start_frame + timesteps_to_translate_to_right,
                    end_frame + timesteps_to_translate_to_right,
                )
            )
            labels.append(0)

        max_time = max([traj[1] for traj in trajs])
        min_time = min([traj[0] for traj in trajs])

        # Get data
        d = {}
        for key in self.data_keys:
            shape = (max_time - min_time, *self.data2shape[key])
            d[key] = np.ndarray(
                shape=shape,
                dtype=self.data2dtype[key],
                buffer=self.shm_mem[key].buf,
                offset=(min_time - self.start) * self.data2nbytes[key],
            )

        obs = d["rgb_static"]
        obss = []

        for traj in trajs:
            start = traj[0] - min_time
            end = start + (traj[1] - traj[0])
            obss.append(obs[start:end])

        labels = [l > 0 for l in labels]

        rows += [-1] * (len(labels) - 1)

        return obss, labels, rows
