import os
import pickle
import random

import blosc
import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from play_segmentation.data.babyai.utils import CL2INSTRUCTION_GOTOLOCAL
from play_segmentation.utils.data import BaseDataset


def get_data(config: dict) -> tuple:
    """
    Get data based on the provided configuration.

    Args:
        config (dict): Configuration parameters for the data.

    Returns:
        data: The data based on the provided configuration.

    Raises:
        NotImplementedError: If the environment is not implemented.
    """
    if config["env"] == "calvin":
        return get_data_calvin(config)
    elif config["env"] == "babyai":
        return get_data_babyai(config)
    else:
        raise NotImplementedError("This environment is not implemented")


def get_data_babyai(config: dict) -> tuple:
    """
    Get training and validation datasets for the BabyAI environment.

    Args:
        config (dict): Configuration parameters for the dataset.

    Returns:
        tuple: A tuple containing the training and validation datasets.

    Raises:
        NotImplementedError: If the specified environment is not implemented.

    """
    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
    else:
        raise NotImplementedError(f"This {config['env_name']} is not implemented")

    dataset_file = os.path.join(
        config["dataset_directory"], config["filename"] + "_training.pkl"
    )
    with open(dataset_file, "rb") as file:
        training_dataset = pickle.load(file)

    dataset_file = os.path.join(
        config["dataset_directory"], config["filename"] + "_validation.pkl"
    )
    with open(dataset_file, "rb") as file:
        validation_dataset = pickle.load(file)

    # Map instruction to class
    inst2cl = {inst: cl for cl, inst in cl2inst.items()}

    # Create data splits
    training_dataset = ActionClassificationBabyAI(training_dataset, inst2cl)
    validation_dataset = ActionClassificationBabyAI(validation_dataset, inst2cl)

    # DataLoader
    training_dataset = DataLoader(
        dataset=training_dataset,
        shuffle=True,
        collate_fn=collate_i3d_babyai,
        batch_size=config["batch_size"],
    )
    validation_dataset = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        collate_fn=collate_i3d_babyai,
        batch_size=config["batch_size"],
    )

    return training_dataset, validation_dataset


def get_data_calvin(config: dict) -> tuple:
    """
    Get the training and validation datasets for the Calvin dataset.

    Args:
        config (dict): Configuration parameters for the dataset.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    datasets = {}
    frames2split = {}

    # Training episodes
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
    split2episodes = {"training": training_episodes, "validation": validation_episodes}

    for split in ["training", "validation"]:
        split_episodes = split2episodes[split]

        episode_directory = os.path.join(config["dataset_directory"], split)

        if split == "training":
            filename = config["labelled_data"]
        else:
            filename = "auto_lang_ann.npy"

        with open(
            os.path.join(episode_directory, f"lang_annotations/{filename}"),
            "rb",
        ) as file:
            lang = np.load(file, allow_pickle=True).item()
            frame2row = lang["language"]["task_frames2row"]

        dataset = ActionClassificationCalvin(
            split,
            config["dataset_directory"],
            episodes,
            split_episodes,
            frame2row,
            frames2split,
        )

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=config["batch_size"],
            collate_fn=collate_i3d_calvin,
            num_workers=config["num_workers"],
        )

        datasets[split] = dataloader

    return datasets["training"], datasets["validation"]


def collate_i3d_babyai(
    samples: list[tuple[np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a list of samples for the I3D BabyAI model.

    Args:
        samples (list): List of tuples, where each tuple contains a video and its corresponding class label.

    Returns:
        tuple: A tuple containing the collated videos and their corresponding class labels.
    """
    videos = []
    video_lengths = []
    classes = []
    states = []

    for video, state, cls in samples:
        videos.append(torch.from_numpy(video).float())
        video_lengths.append(video.shape[0])
        states.append(torch.from_numpy(state).float())
        classes.append(cls)

    video_lengths = torch.tensor(video_lengths, dtype=torch.long)
    videos = pad_sequence(videos, batch_first=True)
    states = pad_sequence(states, batch_first=True)
    classes = torch.tensor(classes, dtype=torch.long)

    return videos, video_lengths, states, classes


def collate_i3d_calvin(
    data: dict,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """
    Collates the data for the I3D Calvin model.

    Args:
        data (dict): A dictionary containing video data and corresponding labels.

    Returns:
        tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]: A tuple containing the collated videos and labels.
    """
    videos = []
    video_lengths = []
    labels = []
    for video, cl in data:
        labels.append(cl)
        videos.append(torch.from_numpy(video).float())
        video_lengths.append(video.shape[0])

    labels = torch.tensor(labels, dtype=torch.long)
    videos = pad_sequence(videos, batch_first=True)
    video_lengths = torch.tensor(video_lengths, dtype=torch.long)

    return videos, video_lengths, None, labels


class ActionClassificationBabyAI(Dataset):
    """
    Dataset class for action classification in the BabyAI environment.

    Args:
        dataset (dict): The dataset containing images and instructions.
        inst2cl (dict): A dictionary mapping instructions to class labels.

    Returns:
        tuple: A tuple containing the images and the corresponding class label.
    """

    def __init__(self, dataset: dict, inst2cl: dict):
        """
        Initialize the Data class.

        Args:
            dataset (dict): The dataset to be used.
            inst2cl (dict): A dictionary mapping instance IDs to class labels.
        """
        self.dataset = dataset
        self.inst2cl = inst2cl

    def __getitem__(self, index: int) -> tuple:
        """
        Get the item at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the images and the corresponding class label.
        """
        images = self.dataset["images"][index]
        images = blosc.unpack_array(images)
        states = self.dataset["states"][index]
        states = blosc.unpack_array(states)
        instruction = self.dataset["instructions"][index]
        cl = self.inst2cl[instruction]

        return images, states, cl

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.dataset["images"])


class ActionClassificationCalvin(BaseDataset):
    """
    Dataset class for action classification using the Calvin dataset.
    """

    def __init__(
        self,
        split: str,
        dataset_directory: str,
        episodes: np.ndarray,
        split_episodes: np.ndarray,
        frames2row: dict,
        frames2split: dict,
    ) -> None:
        """
        Initialize the Data class.

        Args:
            split (str): The split of the dataset (e.g., 'train', 'val', 'test').
            dataset_directory (str): The directory where the dataset is located.
            episodes (np.ndarray): An array of episode IDs.
            split_episodes (np.ndarray): An array of episode IDs for the current split.
            frames2row (dict): A dictionary mapping frame IDs to row indices.
            frames2split (dict): A dictionary mapping frame IDs to split IDs.
        """
        self.data_keys: list[str] = ["rgb_static"]

        super().__init__(
            split,
            dataset_directory,
            episodes,
            split_episodes,
            frames2row,
            frames2split,
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of frames in the dataset.
        """
        return len(self.frames2row)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the RGB static data and the corresponding label.
        """
        start_frame, end_frame = random.choice(list(self.frames2row.keys()))

        window_size: int = end_frame - start_frame
        row: int = self.frames2row[(start_frame, end_frame)]

        d: dict[str, np.ndarray] = {}
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

        return (d["rgb_static"], row)
