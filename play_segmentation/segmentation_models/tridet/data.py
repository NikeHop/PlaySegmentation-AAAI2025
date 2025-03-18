import os
import pickle
import random

from collections import defaultdict

import blosc
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from play_segmentation.data.babyai.utils import (
    CL2INSTRUCTION_GOTOLOCAL,
)
from play_segmentation.segmentation_models.data import SegmentationBaseDataset


def get_data_tridet(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Get the data for the Tridet model based on the given configuration.

    Args:
        config (dict): Configuration parameters for the data.

    Returns:
        data: The data for the Tridet model.

    Raises:
        NotImplementedError: If the specified environment is not implemented.
    """
    if config["env"] == "babyai":
        return get_data_tridet_babyai(config)
    elif config["env"] == "calvin":
        return get_data_tridet_calvin(config)
    else:
        raise NotImplementedError(
            f"This environment: {config['env']} is not implemented"
        )


def get_data_tridet_babyai(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Get the training and validation data loaders for the TriDet model in the BabyAI environment.

    Args:
        config (dict): Configuration parameters for the data loading.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
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

    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
    else:
        raise NotImplementedError(
            f"This environment: {config['env_name']} is not implemented"
        )

    inst2cl = {value: key for key, value in cl2inst.items()}

    training_dataset = TriDetDatasetBabyAI(
        training_dataset, config["window_size"], inst2cl
    )
    validation_dataset = TriDetDatasetBabyAI(
        validation_dataset, config["window_size"], inst2cl
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset=training_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=collate_tridet,
        num_workers=config["num_workers"],
    )
    val_dataloader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=collate_tridet,
        num_workers=config["num_workers"],
    )

    return train_dataloader, val_dataloader


def get_data_tridet_calvin(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Retrieves the training and validation datasets for the TriDet model.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.
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

        dataset = TriDetDatasetCalvin(
            split,
            config["dataset_directory"],
            episodes,
            split2episodes[split],
            frames2row,
            frames2split,
            config["buffer"],
        )

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=config["batch_size"],
            collate_fn=collate_tridet,
            num_workers=config["num_workers"],
        )

        datasets[split] = dataloader

    return datasets["training"], datasets["validation"]


def collate_tridet(data: list) -> tuple:
    """
    Collates a list of samples into a batch.

    Args:
        data (list): A list of samples, where each sample is a tuple containing the following:
            - video (numpy.ndarray): The video data.
            - label (list): The label data.
            - gt_offset (list): The ground truth offset data.

    Returns:
        tuple: A tuple containing the following:
            - videos (torch.Tensor): The batch of video data.
            - obs_lengths (torch.Tensor): The lengths of the videos in the batch.
            - gt_offsets (torch.Tensor): The batch of ground truth offset data.
            - labels (torch.Tensor): The batch of label data.
    """
    videos = []
    obs_lengths = []
    gt_offsets = []
    labels = []

    for sample in data:
        if sample is None:
            continue
        videos.append(torch.from_numpy(sample[0]).float())
        obs_lengths.append(videos[-1].shape[0])
        labels.append(torch.tensor(sample[1], dtype=torch.long).reshape(1, -1))
        gt_offsets.append(
            torch.tensor(
                [sample[2][1] - sample[2][0], sample[2][2] - sample[2][0]],
                dtype=torch.float,
            ).reshape(1, -1)
        )
    videos = pad_sequence(videos, batch_first=True)
    obs_lengths = torch.tensor(obs_lengths, dtype=torch.long)

    return videos, obs_lengths, gt_offsets, labels


class TriDetDatasetBabyAI(Dataset):
    """
    Dataset class for TriDet model with BabyAI dataset.

    Args:
        dataset (dict): The dataset containing images, actions, rewards, directions, instructions, and instruction types.
        window_size (int): The desired window size for segmenting the trajectories.
        inst2cl (dict): A dictionary mapping instructions to class labels.
        image_data (str): The key for accessing the image data in the dataset.

    Attributes:
        dataset (dict): The filtered dataset containing images, actions, rewards, directions, instructions, and instruction types.
        inst2cl (dict): A dictionary mapping instructions to class labels.
        image_data (str): The key for accessing the image data in the dataset.
        window_size (int): The desired window size for segmenting the trajectories.
        length (int): The length of the dataset.

    Methods:
        __getitem__(self, index): Retrieves a specific item from the dataset.
        __len__(self): Returns the length of the dataset.
        filter_dataset(self, dataset, window_size): Filters the dataset based on the window size.
        get_length(self, index): Returns the length of a specific trajectory in the dataset.
    """

    def __init__(
        self, dataset: dict, window_size: int, inst2cl: dict, image_data: str = "images"
    ) -> None:
        super().__init__()

        self.dataset = self.filter_dataset(dataset, window_size)
        self.inst2cl = inst2cl
        self.image_data = image_data
        self.window_size = window_size
        self.length = len(self.dataset["images"])

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves a specific item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the video segment, class label, and time range of the segment.
        """
        # Determine the length of the different segments
        valid_segments = []
        unseg_traj = self.dataset["images"][index]
        segments = [blosc.unpack_array(elem)[:-1] for elem in unseg_traj[:-1]] + [
            blosc.unpack_array(unseg_traj[-1])
        ]
        obs_seq = np.concatenate(segments, axis=0)
        for i, segment in enumerate(segments):
            if segment.shape[0] < self.window_size:
                valid_segments.append(i)

        if len(valid_segments) == 0:
            return None

        # Sample a segment
        segment_index = random.choice(valid_segments)
        timesteps = np.array([0] + [segment.shape[0] for segment in segments]).cumsum()
        start_time = timesteps[segment_index]
        end_time = timesteps[segment_index + 1]
        length = timesteps[-1]
        instruction = self.dataset["instructions"][index][segment_index]
        cl = self.inst2cl[instruction]

        segment_size = end_time - start_time
        max_steps_to_right = length - end_time
        max_steps_to_left = start_time

        if max_steps_to_left > max_steps_to_right:
            steps_to_right = max_steps_to_right
            steps_to_left = self.window_size - segment_size - steps_to_right
        else:
            steps_to_left = max_steps_to_left
            steps_to_right = self.window_size - segment_size - steps_to_left

        # Get video_segment
        video_segment = obs_seq[start_time - steps_to_left : end_time + steps_to_right]

        return (
            video_segment,
            cl,
            (
                start_time - steps_to_left,
                start_time,
                end_time,
                end_time + steps_to_right,
            ),
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.length

    def filter_dataset(self, dataset: dict, window_size: int) -> dict:
        """
        Filters the dataset based on the window size.

        Args:
            dataset (dict): The dataset to filter.
            window_size (int): The desired window size.

        Returns:
            dict: The filtered dataset.
        """
        new_dataset = defaultdict(list)
        for i in range(len(dataset["images"])):
            traj_length = sum([len(actions) for actions in dataset["actions"][i]])
            if traj_length > window_size:
                new_dataset["images"].append(dataset["images"][i])
                new_dataset["actions"].append(dataset["actions"][i])
                new_dataset["rewards"].append(dataset["rewards"][i])
                new_dataset["directions"].append(dataset["directions"][i])
                new_dataset["instructions"].append(dataset["instructions"][i])
                new_dataset["instruction_types"].append(dataset["instruction_types"][i])

        return new_dataset

    def get_length(self, index: int) -> int:
        """
        Returns the length of a specific trajectory in the dataset.

        Args:
            index (int): The index of the trajectory.

        Returns:
            int: The length of the trajectory.
        """
        return sum([len(actions) for actions in self.dataset["actions"][index]])


class TriDetDatasetCalvin(SegmentationBaseDataset):
    """
    A dataset class for TriDet segmentation model with Calvin dataset.

    Args:
        split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        dataset_directory (str): The directory path of the dataset.
        episodes (list): List of episode names.
        split_episodes (dict): Dictionary mapping split names to episode names.
        frames2row (dict): Dictionary mapping frame tuples to row indices.
        frames2split (dict): Dictionary mapping frame tuples to split names.
        buffer (int): Buffer size.
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
    ):
        self.data_keys: list[str] = ["rgb_static"]
        self.window_size: int = 128

        super().__init__(
            split,
            dataset_directory,
            episodes,
            split_episodes,
            frames2row,
            frames2split,
            buffer,
            ["rgb_static"],
        )

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the item at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the RGB static data, row index, and frame information.
        """
        original_start_frame, original_end_frame = random.choice(
            list(self.frames2row.keys())
        )

        start_episode, end_episode = self.frames2episode[
            (original_start_frame, original_end_frame)
        ]

        # Sample displacement
        max_start_displacement = original_start_frame - start_episode
        max_end_displacement = end_episode - original_end_frame
        fraction_left = random.random()
        distance_remainder = self.window_size - (
            original_end_frame - original_start_frame
        )
        distance_left = int(distance_remainder * fraction_left)
        distance_right = distance_remainder - distance_left

        leftover = 0
        if max_start_displacement > 0:
            start_displacement = min(max_start_displacement, distance_left)
            leftover += distance_left - start_displacement
        else:
            start_displacement = 0
            leftover += distance_left

        if max_end_displacement > 0:
            end_displacement = min(max_end_displacement, distance_right)
            leftover += distance_right - end_displacement
        else:
            end_displacement = 0
            leftover += distance_right

        if end_displacement == max_end_displacement:
            start_displacement += leftover

        if start_displacement == max_start_displacement:
            end_displacement += leftover

        start_frame = original_start_frame - start_displacement
        end_frame = original_end_frame + end_displacement
        window_size = end_frame - start_frame

        row = self.frames2row[(original_start_frame, original_end_frame)]

        d = {}
        for key in self.data_keys:
            shape = (
                window_size,
                *self.data2shape[key],
            )
            d[key] = np.ndarray(
                shape=shape,
                dtype=self.data2dtype[key],
                buffer=self.shm_mem[key].buf,
                offset=(start_frame - self.start) * self.data2nbytes[key],
            )

        return (
            d["rgb_static"],
            row,
            (start_frame, original_start_frame, original_end_frame, end_frame),
        )
