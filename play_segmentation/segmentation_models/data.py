import random

import blosc
import numpy as np

from torch.utils.data import Dataset

from play_segmentation.data.babyai.generate import InstructionType
from play_segmentation.utils.data import BaseDataset


class BabyAIBaseDataset(Dataset):
    """
    Dataset class for BabyAI segmentation models.

    Args:
        dataset (dict): The dataset containing the trajectory data.
        max_steps (int): The maximum number of steps for each segment.
        inst2cl (dict): A dictionary mapping instruction types to class labels.
        image_data (str): The key for accessing the image data in the dataset.

    Attributes:
        dataset (dict): The dataset containing the trajectory data.
        max_steps (int): The maximum number of steps for each segment.
        inst2cl (dict): A dictionary mapping instruction types to class labels.
        image_data (str): The key for accessing the image data in the dataset.
    """

    def __init__(self, dataset, max_steps, inst2cl, image_data="images"):
        super().__init__()

        self.dataset = dataset
        self.max_steps = max_steps
        self.inst2cl = inst2cl
        self.image_data = image_data

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the video segment, class label, and segment information.
        """
        # Sample an unsegmented trajectory
        unseg_traj = self.dataset[self.image_data][index]
        instruction_types = self.dataset["instruction_types"][index]

        # Unpack trajectory and transform it into array
        segments = [blosc.unpack_array(elem) for elem in unseg_traj]
        obs_seq = np.concatenate(segments, axis=0)

        # Determine length of unsegmented trajectory
        length = obs_seq.shape[0]
        timesteps = np.array([0] + [segment.shape[0] for segment in segments]).cumsum()

        # Sample a segment from the unsegmented trajectory
        index = random.randint(0, len(unseg_traj) - 1)
        start_time = timesteps[index]
        end_time = timesteps[index + 1] - 1
        instruction_type = InstructionType()
        instruction_type.from_dict(instruction_types[index])
        instruction = str(instruction_type)
        cl = self.inst2cl[instruction]

        # Check special cases
        if start_time == 0:
            steps_to_left = 0
            if min(self.max_steps, length - end_time - 1) > 0:
                steps_to_right = random.randint(
                    1, min(self.max_steps, length - end_time - 1)
                )
            else:
                steps_to_right = 0

        elif end_time == length - 1:
            if min(self.max_steps, start_time) > 0:
                steps_to_left = random.randint(1, min(self.max_steps, start_time))
            else:
                steps_to_left = 0
            steps_to_right = 0

        else:
            if min(self.max_steps, start_time) > 0:
                steps_to_left = random.randint(1, min(self.max_steps, start_time))
            else:
                steps_to_left = 0
            if min(self.max_steps, length - end_time - 1) > 0:
                steps_to_right = random.randint(
                    1, min(self.max_steps, length - end_time - 1)
                )
            else:
                steps_to_left = 0

        # Get video_segment
        video_segment = obs_seq[
            start_time - steps_to_left : end_time + steps_to_right + 1
        ]
        return video_segment, cl, (start_time, end_time, steps_to_left, steps_to_right)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset["instructions"])


class SegmentationBaseDataset(BaseDataset):
    """
    A base dataset class for segmentation models.

    Args:
        split (str): The split of the dataset.
        dataset_directory (str): The directory where the dataset is stored.
        episodes (list): List of episodes.
        split_episodes (dict): Dictionary mapping split names to episode names.
        frames2row (dict): Dictionary mapping frame tuples to instruction class.
        frames2split (dict): Dictionary mapping frame tuples to split names.
        buffer (int): The buffer size for displacement sampling.
        keys (list): List of data keys.

    Attributes:
        data_keys (list): List of data keys.
        buffer (int): The buffer size for displacement sampling.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Retrieves an item from the dataset.

    """

    def __init__(
        self,
        split,
        dataset_directory,
        episodes,
        split_episodes,
        frames2row,
        frames2split,
        buffer,
        keys,
    ):
        self.data_keys = keys

        super().__init__(
            split,
            dataset_directory,
            episodes,
            split_episodes,
            frames2row,
            frames2split,
        )

        self.buffer = buffer

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.

        """
        return len(self.frames2row)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the data, row index, and frame information.

        """
        original_start_frame, original_end_frame = random.choice(
            list(self.frames2row.keys())
        )
        start_episode, end_episode = self.frames2episode[
            (original_start_frame, original_end_frame)
        ]

        # Sample displacement
        max_start_displacement = min(self.buffer, original_start_frame - start_episode)
        max_end_displacement = min(self.buffer, end_episode - original_end_frame)

        if max_start_displacement > 0:
            start_displacement = random.sample(range(0, max_start_displacement), k=1)[0]
        else:
            start_displacement = 0

        if max_end_displacement > 0:
            end_displacement = random.sample(range(0, max_end_displacement), k=1)[0]
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
            d,
            row,
            (start_frame, original_start_frame, original_end_frame, end_frame),
        )
