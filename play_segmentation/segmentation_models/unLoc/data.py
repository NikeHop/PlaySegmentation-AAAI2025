import os
import pickle
import random

from typing import Callable

import blosc
import clip
import numpy as np
import torch
import tqdm

from einops import repeat, rearrange
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from play_segmentation.segmentation_models.data import (
    SegmentationBaseDataset,
)
from play_segmentation.data.babyai.utils import CL2INSTRUCTION_GOTOLOCAL


def get_data_unloc(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Get the data for the UnLoc model based on the provided configuration.

    Args:
        config (dict): The configuration for the data.

    Returns:
        Tuple[DataLoader, DataLoader]: The data for the unLoc model.

    Raises:
        NotImplementedError: If the specified dataset is not implemented.
    """
    if config["dataset"] == "babyai":
        return get_data_unloc_babyai(config)
    elif config["dataset"] == "calvin":
        return get_data_unloc_calvin(config)
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} not implemented")


def get_data_unloc_babyai(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Get the training and validation dataloaders for the BabyAIUnLoc dataset.

    Args:
        config (dict): Configuration parameters for the dataset.

    Returns:
        tuple: A tuple containing the training and validation dataloaders.
    """
    # Instruction-Class mapping
    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
    inst2cl = {value: key for key, value in cl2inst.items()}

    # Transform dataset to CLIP embeddings
    training_dataset = config["dataset_file"] + "_training.pkl"
    validation_dataset = config["dataset_file"] + "_validation.pkl"
    training_clip_dataset = config["dataset_file"] + "_training_clip.pkl"
    validation_clip_dataset = config["dataset_file"] + "_validation_clip.pkl"

    if not os.path.exists(
        os.path.join(config["dataset_directory"], training_clip_dataset)
    ):
        training_clip_dataset = transform2clip(training_dataset, cl2inst, config)
    else:
        with open(
            os.path.join(config["dataset_directory"], training_clip_dataset), "rb"
        ) as file:
            training_clip_dataset = pickle.load(file)

    if not os.path.exists(
        os.path.join(config["dataset_directory"], validation_clip_dataset)
    ):
        validation_clip_dataset = transform2clip(validation_dataset, cl2inst, config)
    else:
        with open(
            os.path.join(config["dataset_directory"], validation_clip_dataset), "rb"
        ) as file:
            validation_clip_dataset = pickle.load(file)

    instruction_embeddings = training_clip_dataset["instruction_embeddings"]

    # Create Datasets and Dataloaders
    training_dataset = BabyAIUnLoc(training_clip_dataset, config["max_step"], inst2cl)
    validation_dataset = BabyAIUnLoc(
        validation_clip_dataset, config["max_step"], inst2cl
    )

    train_dataloader = DataLoader(
        training_dataset,
        collate_fn=collate_babyai_decorator(instruction_embeddings),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        validation_dataset,
        collate_fn=collate_babyai_decorator(instruction_embeddings),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    return train_dataloader, val_dataloader


def transform2clip(dataset_file: str, cl2inst: dict, config: dict) -> dict:
    """
    Transforms the dataset by encoding images and instructions using the CLIP model.

    Args:
        dataset_file (str): The path to the dataset file.
        cl2inst (dict): A dictionary mapping class labels to instructions.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        dict: The transformed dataset.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """
    # Load clip model
    model, transforms = clip.load(name=config["vlm"], device=config["device"])

    # Load dataset
    with open(os.path.join(config["dataset_directory"], dataset_file), "rb") as file:
        dataset = pickle.load(file)

    clip_unseg_trajs = []
    with torch.no_grad():
        # Encode images
        for unseg_traj in tqdm.tqdm(dataset["images"]):
            clip_trajs = []
            for traj in unseg_traj:
                traj = [
                    transforms(Image.fromarray(elem))
                    for elem in blosc.unpack_array(traj)
                ]
                traj = torch.stack(traj, dim=0).to(config["device"])
                clip_traj = model.encode_image(traj)
                clip_traj = clip_traj.cpu().numpy()
                clip_traj = blosc.pack_array(clip_traj)
                clip_trajs.append(clip_traj)
            clip_unseg_trajs.append(clip_trajs)

        dataset["clip_images"] = clip_unseg_trajs

        # Encode instructions
        instructions = list(cl2inst.values())
        tokenized_instructions = clip.tokenize(instructions).to(config["device"])
        n_instructions = len(instructions)
        n_encoded_instructions = 0
        encoded_instructions = []
        bs = 100
        pbar = tqdm.tqdm(total=n_instructions)
        while n_encoded_instructions < n_instructions:
            batch_tokenized_instructions = tokenized_instructions[
                n_encoded_instructions : n_encoded_instructions + bs
            ]
            n_encoded_instructions += bs
            batch_encoded_instructions = model.encode_text(batch_tokenized_instructions)
            encoded_instructions.append(batch_encoded_instructions)
            pbar.update(bs)
        encoded_instructions = torch.cat(encoded_instructions, dim=0)
        encoded_instructions = encoded_instructions

        inst2clip_embed = {}
        for enc_instruction, instruction in zip(encoded_instructions, instructions):
            inst2clip_embed[instruction] = enc_instruction

        instruction_embeddings = []
        for cl in range(len(cl2inst)):
            instruction_embeddings.append(inst2clip_embed[cl2inst[cl]])
        instruction_embeddings = torch.stack(instruction_embeddings, dim=0).cpu()

        dataset["instruction_embeddings"] = instruction_embeddings

    clip_dataset = dataset_file.replace(".pkl", "_clip.pkl")
    with open(os.path.join(config["dataset_directory"], clip_dataset), "wb") as file:
        pickle.dump(dataset, file)

    return dataset


def get_data_unloc_calvin(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Retrieves and prepares the Calvin dataset for training and validation of UnLoc.

    Args:
        config (dict): Configuration parameters for the dataset.

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
            instruction_embeddings = (
                torch.from_numpy(lang["language"]["clip"]).float().squeeze(1)
            )

            frames2row = {
                frame: lang["language"]["clip_frames2row"][frame]
                for frame in lang["info"]["indx"]
            }

        dataset = UnlocDataCalvin(
            split,
            config["dataset_directory"],
            episodes,
            split2episodes[split],
            frames2row,
            frames2split,
            config["buffer"],
            config["window_size"],
        )

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=config["batch_size"],
            collate_fn=create_collate_unloc(instruction_embeddings),
            num_workers=config["num_workers"],
        )

        datasets[split] = dataloader

    return datasets["training"], datasets["validation"]


def create_collate_unloc(instruction_embeddings: torch.Tensor) -> Callable:
    """
    Create a collate function for the UnLoc dataset.

    Args:
        instruction_embeddings (torch.Tensor): Embeddings of instructions.

    Returns:
        collate_unloc (function): Collate function for the unLoc dataset.
    """
    num_classes, D = instruction_embeddings.shape
    bg_cl = num_classes
    num_classes += 1
    instruction_embeddings = torch.cat(
        [instruction_embeddings, torch.zeros(1, D)], dim=0
    )

    def collate_unloc(data: list) -> dict:
        """
        Collates a list of samples into a batch for the UnLoc model.

        Args:
            data (list): List of samples, where each sample is a tuple containing the following elements:
                - sample[0]: Clip frame tensor of shape (T, D), where T is the number of frames and D is the feature dimension.
                - sample[1]: Class label for the sample.
                - sample[2]: Tuple containing the start and end indices of the instruction in the clip frame tensor.

        Returns:
            dict: A dictionary representing the batch, with the following keys:
                - "clip_seq": Padded clip frame tensor of shape (B, T, C, D), where B is the batch size, T is the maximum number of frames,
                  C is the number of classes, and D is the feature dimension.
                - "classes": Padded class label tensor of shape (B, max_cols), where B is the batch size and max_cols is the maximum number of columns in the classes tensor.
                - "obs_lengths": Tensor of shape (B,) containing the actual number of frames in each sample.
                - "img_obs_lengths": Tensor of shape (B,) containing the actual number of frames in each sample's clip frame tensor.
                - "mask": Boolean tensor of shape (B, max_cols) indicating the valid elements in the classes tensor.
        """
        clip_frames = []
        obs_lengths = []
        img_obs_lengths = []
        classes = []
        masks = []

        for i, sample in enumerate(data):
            clip_frame = torch.tensor(sample[0], dtype=torch.float)  # TxD
            img_obs_lengths.append(clip_frame.shape[0])
            clip_frame = repeat(clip_frame, "t d -> t c d", c=num_classes)
            clip_frame = torch.cat(
                [clip_frame, instruction_embeddings.unsqueeze(0)], dim=0
            )
            clip_frames.append(clip_frame)
            obs_lengths.append(clip_frame.shape[0])
            front_buffer = sample[2][1] - sample[2][0]
            instruction_length = sample[2][2] - sample[2][1]
            end_buffer = sample[2][3] - sample[2][2]
            cl = torch.tensor(
                [bg_cl] * front_buffer
                + [sample[1]] * instruction_length
                + [bg_cl] * end_buffer,
                dtype=torch.long,
            )

            classes.append(cl)
            masks.append(cl.shape[0])

        clip_seq = pad_sequence(clip_frames, batch_first=True)  # B,T,C,D
        clip_seq = rearrange(clip_seq, "b t c d -> b c t d")

        obs_lengths = torch.tensor(obs_lengths, dtype=torch.long)
        img_obs_lengths = torch.tensor(img_obs_lengths, dtype=torch.long)
        classes = pad_sequence(classes, batch_first=True)
        max_cols = classes.shape[-1]

        mask = []
        for i, elem in enumerate(masks):
            mask += [True] * elem
            mask += [False] * (max_cols - elem)
        mask = torch.tensor(mask, dtype=torch.bool)

        batch = {
            "clip_seq": clip_seq,
            "classes": classes,
            "obs_lengths": obs_lengths,
            "img_obs_lengths": img_obs_lengths,
            "mask": mask,
        }

        return batch

    return collate_unloc


def collate_babyai_decorator(instruction_embeddings: torch.Tensor) -> Callable:
    """
    Decorator function that returns a collate function for BabyAI dataset.

    Args:
        instruction_embeddings (torch.Tensor): Tensor containing instruction embeddings.

    Returns:
        Callable: Collate function for BabyAI dataset.
    """
    # Add background instruction
    n_instructions, D = instruction_embeddings.shape
    instruction_embeddings = torch.cat(
        [instruction_embeddings, torch.zeros(1, D)], dim=0
    )
    n_instructions += 1
    background_class = n_instructions - 1

    def collate_babyai(data: list) -> dict:
        """
        Collates a list of samples into a batch for training or evaluation.

        Args:
            data (list): A list of samples, where each sample is a tuple containing:
                - video_sequence (torch.Tensor): The video sequence.
                - cl (int): The instruction class.
                - start_time (int): The start time of the video segment.
                - end_time (int): The end time of the video segment.
                - steps_to_left (int): The number of steps to the left.
                - steps_to_right (int): The number of steps to the right.

        Returns:
            dict: A dictionary containing the batched data with the following keys:
                - "clip_seq" (torch.Tensor): The batched video sequences with instructions.
                - "classes" (torch.Tensor): The batched instruction classes.
                - "obs_lengths" (torch.Tensor): The lengths of each video sequence with instructions.
                - "img_obs_lengths" (torch.Tensor): The lengths of each video sequence without instructions.
                - "mask" (torch.Tensor): The mask indicating the valid elements in the batched instruction classes.
        """
        clip_seqs = []
        obs_lengths = []
        img_obs_lengths = []
        classes = []
        masks = []

        for sample in data:
            video_sequence = sample[0]
            cl = sample[1]
            start_time, end_time, steps_to_left, steps_to_right = sample[2]

            # Repeat the video for each instruction class
            video_sequence = torch.tensor(video_sequence, dtype=torch.float)
            video_sequence = repeat(video_sequence, "t d -> t c d", c=n_instructions)
            img_obs_lengths.append(video_sequence.shape[0])

            # Append the instruction
            clip_frames = torch.cat(
                [video_sequence, instruction_embeddings.unsqueeze(0)], dim=0
            )
            obs_lengths.append(clip_frames.shape[0])
            clip_seqs.append(clip_frames)

            # Create framewise labels
            cl = torch.tensor(
                [background_class] * steps_to_left
                + [cl] * (end_time - start_time)
                + [background_class] * steps_to_right,
                dtype=torch.long,
            )

            classes.append(cl)
            masks.append(cl.shape[0])

        clip_seqs = pad_sequence(clip_seqs, batch_first=True)
        clip_seqs = rearrange(clip_seqs, "b t c d -> b c t d")
        obs_lengths = torch.tensor(obs_lengths, dtype=torch.long)
        img_obs_lengths = torch.tensor(img_obs_lengths, dtype=torch.long)
        classes = pad_sequence(classes, batch_first=True)
        max_cols = classes.shape[-1]
        mask = []

        for i, elem in enumerate(masks):
            mask += [True] * elem
            mask += [False] * (max_cols - elem)
        mask = torch.tensor(mask, dtype=torch.bool)

        batch = {
            "clip_seq": clip_seqs,
            "classes": classes,
            "obs_lengths": obs_lengths,
            "img_obs_lengths": img_obs_lengths,
            "mask": mask,
        }

        return batch

    return collate_babyai


class BabyAIUnLoc(Dataset):
    """
    Dataset class for unlocalized trajectory segmentation in the BabyAI environment.

    Args:
        dataset (dict): The dataset containing the trajectory data.
        max_steps (int): The maximum number of steps to consider for segmenting the trajectory.
        inst2cl (dict): A dictionary mapping instructions to class labels.

    Attributes:
        dataset (dict): The dataset containing the trajectory data.
        max_steps (int): The maximum number of steps to consider for segmenting the trajectory.
        inst2cl (dict): A dictionary mapping instructions to class labels.
        length (int): The total number of segments in the dataset.
    """

    def __init__(self, dataset: dict, max_steps: int, inst2cl: dict) -> None:
        super().__init__()

        self.dataset = dataset
        self.max_steps = max_steps
        self.inst2cl = inst2cl
        self.length = sum([len(elem) for elem in self.dataset["instruction_types"]])

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the video segment, class label, and additional information.
                - video_segment (numpy.ndarray): The video segment.
                - cl (str): The class label.
                - additional_info (tuple): Additional information about the segment, including start time,
                  end time, steps to the left, and steps to the right.
        """
        index = random.randint(0, len(self.dataset["instructions"]) - 1)

        # Sample an unsegmented trajectory
        unseg_traj = self.dataset["clip_images"][index]
        instructions = self.dataset["instructions"][index]

        # Unpack trajectory and transform it into array
        segments = [blosc.unpack_array(elem)[:-1] for elem in unseg_traj[:-1]] + [
            blosc.unpack_array(unseg_traj[-1])
        ]
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

        # Check special cases
        if start_time == 0:
            steps_to_left = 0
            if (length - end_time - 1) > 0:
                steps_to_right = random.randint(
                    1, min(self.max_steps, length - end_time - 1)
                )
            else:
                steps_to_right = 0

        elif end_time == length:
            steps_to_right = 0
            if start_time > 0:
                steps_to_left = random.randint(1, min(self.max_steps, start_time))
            else:
                steps_to_left = 0

        else:
            steps_to_left = random.randint(1, min(self.max_steps, start_time))
            steps_to_right = random.randint(
                1, min(self.max_steps, length - end_time - 1)
            )

        # Get video_segment
        video_segment = obs_seq[start_time - steps_to_left : end_time + steps_to_right]
        return video_segment, cl, (start_time, end_time, steps_to_left, steps_to_right)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.length


class UnlocDataCalvin(SegmentationBaseDataset):
    """
    Dataset class for UNLOC data with Calvin's modifications.

    Args:
        split (str): The split of the dataset.
        dataset_directory (str): The directory of the dataset.
        episodes (np.ndarray): Array of episode indices.
        split_episodes (np.ndarray): Array of split episode indices.
        frames2row (dict): Dictionary mapping frames to row indices.
        frames2split (dict): Dictionary mapping frames to split indices.
        buffer (dict): Dictionary of buffer data.
        window_size (int): The size of the window.
        debug (bool): Flag indicating whether to run in debug mode.
    """

    def __init__(
        self,
        split: str,
        dataset_directory: str,
        episodes: np.ndarray,
        split_episodes: np.ndarray,
        frames2row: dict,
        frames2split: dict,
        buffer: dict,
        window_size: int,
    ):
        self.data_keys: list[str] = [
            "clip_pretrained",
        ]

        self.window_size: int = window_size

        super().__init__(
            split,
            dataset_directory,
            episodes,
            split_episodes,
            frames2row,
            frames2split,
            buffer,
            ["clip_pretrained"],
        )

        self.buffer: dict = buffer

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the clip_pretrained data, row index, and frame information.
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

        if max_start_displacement > 0:
            start_displacement = min(max_start_displacement, distance_left)
        else:
            start_displacement = 0

        if max_end_displacement > 0:
            end_displacement = min(max_end_displacement, distance_right)
        else:
            end_displacement = 0

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
            d["clip_pretrained"],
            row,
            (start_frame, original_start_frame, original_end_frame, end_frame),
        )

    @property
    def data_keys(self) -> None:
        """
        Get the data keys.

        Returns:
            list: The data keys.
        """
        return self._data_keys

    @data_keys.setter
    def data_keys(self, value: list[str]) -> None:
        """
        Setter method for the data_keys property.

        Args:
            value (list): The list of data keys to be set.

        Returns:
            None
        """
        self._data_keys = value
