# Utilities related to datasets needed across different algorithms

import multiprocessing as mp
import os
import pickle
import time

from abc import ABC, abstractmethod
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import tqdm

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Base class for CALVIN datasets.

    Args:
        split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        dataset_directory (str): The directory where the dataset is stored.
        episodes (list): List of episodes, each represented as a tuple of start and end frames.
        split_episodes (dict): Dictionary mapping frames to their corresponding split.
        frames2row (dict): Dictionary mapping frames to their corresponding row in the dataset.
        frames2split (dict): Dictionary mapping frames to their corresponding split.
        debug (bool): Flag indicating whether to enable debug mode.

    Attributes:
        split (str): The split of the dataset.
        waiting_time (int): The maximum waiting time for data to be loaded.
        dataset_directory (str): The directory where the dataset is stored.
        frames2row (dict): Dictionary mapping frames to their corresponding instruction class in the dataset.
        episodes (list): List of episodes, each represented as a tuple of start and end frames.
        split_episodes (dict): Dictionary mapping frames to their corresponding split.
        window_size_low (int): The lower bound of the window size.
        window_size_high (int): The upper bound of the window size.
        data2itemsize (dict): Dictionary mapping data keys to their corresponding item size.
        data2shape (dict): Dictionary mapping data keys to their corresponding shape.
        data2dtype (dict): Dictionary mapping data keys to their corresponding data type.
        data2nbytes (dict): Dictionary mapping data keys to their corresponding number of bytes.
        dataset_size (int): The total number of timesteps in the dataset.
        start (int): The start frame of the dataset.
        end (int): The end frame of the dataset.
        frames2episode (dict): Dictionary mapping frames to their corresponding episode.
        shm_mem (dict): Dictionary of shared memory objects.
        data_keys (list): List of keys for the data to be loaded.
        data (dict): Dictionary of loaded data.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the item at the specified index.
        _prepare_data_statistics(): Prepares statistics for the data.
        _prepare_data(): Prepares the data for loading.
        _get_frame(idx): Retrieves the data for a given frame index.
        _get_filename(idx): Generates the filename for a given frame index.
        _get_sh_mem(): Loads or creates shared memory for the data.
        _loading_done(): Signals that the data loading is complete.
        _check_shm_mem_existence(): Checks if shared memory exists.
        _load_shm_mem(): Loads shared memory if it exists.
        _create_sh_mem(): Creates shared memory for the data.
        _free_shm_mem(keys): Frees shared memory for the specified keys.
        _worker_process(frames, shm_mem): Worker process for loading data into shared memory.
        _load_data(): Loads the data into shared memory.
    """

    def __init__(
        self,
        split: str,
        dataset_directory: str,
        episodes: np.ndarray,
        split_episodes: np.ndarray,
        frames2row: dict[int, int],
        frames2split: dict[int, str],
    ) -> None:
        self.split = split
        self.waiting_time = 7200  # Time to wait for Construction of SharedMemory
        self.dataset_directory = dataset_directory
        self.frames2row = frames2row
        self.episodes = list(sorted(episodes, key=lambda x: x[0]))
        self.split_episodes = split_episodes

        self.frames2split = frames2split

        self.window_size_low = 32
        self.window_size_high = 64
        self._prepare_data()

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

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def _prepare_data_statistics(self) -> None:
        """
        Prepare data statistics.
        This method calculates various info for the data to be loaded.

        Returns:
            None
        """
        frame_data = self._get_frame(self.episodes[0][0])
        self.data2itemsize = {}
        self.data2shape = {}
        self.data2dtype = {}
        self.data2nbytes = {}
        for key, value in frame_data.items():
            if key in self.data_keys:
                self.data2itemsize[key] = value.itemsize
                self.data2shape[key] = value.shape
                self.data2dtype[key] = value.dtype
                self.data2nbytes[key] = value.nbytes

    def _prepare_data(self) -> None:
        """
        Prepare the data for segmentation.

        This method computes the dataset size, smallest and largest timestamps,
        and maps each frame to its corresponding episode. It also prepares data statistics
        and gets shared memory.

        Returns:
            None
        """
        # Dataset size == Number of timesteps summed over all episodes
        self.dataset_size = sum([frames[1] - frames[0] for frames in self.episodes])
        self.start = min([frames[0] for frames in self.episodes])
        self.end = max([frames[1] for frames in self.episodes])

        # Compute for each segment the corresponding episode
        self.frames2episode = {
            frame: get_episode(frame, self.episodes) for frame in self.frames2row.keys()
        }

        self._prepare_data_statistics()
        self._get_sh_mem()

    def _get_frame(self, idx: int) -> np.lib.npyio.NpzFile:
        """
        Get the frame data for a given index.

        Args:
            idx (int): The index of the frame.

        Returns:
            np.lib.npyio.NpzFile: The frame data.

        Raises:
            Exception: If there is an error loading the frame data.
        """
        idx = "0" * (7 - len(str(idx))) + str(idx)
        # Is it a training frame or a validation frame
        filename = self._get_filename(idx)
        try:
            data = np.load(filename, allow_pickle=True)
        except Exception as e:
            print(f"Error in {idx}: {e}")
            idx = "0053819"
            filename = os.path.join(
                self.dataset_directory, self.frames2split[53819], f"episode_{idx}.npz"
            )
            data = np.load(filename)

        return data

    def _get_filename(self, idx: int) -> str:
        """
        Get the filename for the given index.

        Args:
            idx (int): The index of the frame.

        Returns:
            str: The filename of the frame.
        """
        split = self.frames2split[idx]
        filename = os.path.join(self.dataset_directory, split, f"episode_{idx}.npz")
        return filename

    def _get_sh_mem(self) -> None:
        """
        Retrieves or creates shared memory for storing data.

        Returns:
            None
        """
        redo_shm_mem = True
        create_shm_mem = True
        keys = self.data_keys

        # Has shm_mem already been loaded
        create_shm_mem = self._check_shm_mem_existence()
        if not create_shm_mem:
            redo_shm_mem = self._load_shm_mem()

        # If shm_mem has to be redone or created
        print(f"redo_shm_mem {redo_shm_mem} create_shm_mem {create_shm_mem}")
        if redo_shm_mem or create_shm_mem:
            self._free_shm_mem(keys)
            self.shm_mem = self._create_sh_mem()
            self._load_data()
            self._loading_done()

    def _loading_done(self):
        print("Loading Done")
        with open(f"/tmp/shm_mem_calvin_loaded.pkl", "wb+") as file:
            pickle.dump({True}, file)

    def _check_shm_mem_existence(self) -> bool:
        """
        Checks the existence of shared memory for the specified data keys.
        If the shared memory exists, it waits for the data to be loaded.
        If the shared memory does not exist, it indicates that new shared memory needs to be created.

        Returns:
            create_shm_mem (bool): Indicates whether new shared memory needs to be created.
        """
        data_exists = all([os.path.exists(f"/dev/shm/{k}") for k in self.data_keys])
        print(f"Data exists {data_exists}")
        if data_exists:
            print("Shared Memory Exists")
            # Enter loop that checks when loading is done
            start = time.time()
            print(os.path.exists(f"/tmp/shm_mem_calvin_loaded.pkl"))
            while not os.path.exists(f"/tmp/shm_mem_calvin_loaded.pkl"):
                time.sleep(60)
                print("Waiting for data being loaded")
                if (time.time() - start) > self.waiting_time:
                    raise Exception("Waiting time for data being loaded exceeded")
            create_shm_mem = False
        else:
            create_shm_mem = True
        return create_shm_mem

    def _load_shm_mem(self) -> bool:
        """
        Load shared memory data.

        This method loads the shared memory data for the given keys. If any parts of the shared memory are missing,
        it sets the `redo_shm_mem` flag to True.

        Returns:
            bool: True if any parts of the shared memory are missing, False otherwise.
        """
        redo_shm_mem = False
        self.shm_mem = {}
        for key in self.data2shape.keys():
            try:
                self.shm_mem[key] = SharedMemory(name=f"{key}")
            except FileNotFoundError:
                print("Parts of the shared memory are missing")
                redo_shm_mem = True

        return redo_shm_mem

    def _create_sh_mem(self) -> dict:
        """
        Creates shared memory for the specified data.

        Returns:
            dict: A dictionary containing the shared memory objects for each data key.
        """
        sh_mem = {}

        # Number of frames
        n_frames = self.end - self.start + 1
        for key in self.data2shape.keys():
            sh_mem[key] = SharedMemory(
                create=True,
                size=self.data2nbytes[key] * n_frames,
                name=f"{key}",
            )

        print("Shared memory created")
        return sh_mem

    def _free_shm_mem(self, keys: list) -> None:
        """
        Frees the shared memory associated with the given keys.

        Args:
            keys (list): A list of keys representing the shared memory segments to be freed.

        Returns:
            None
        """
        for key in keys:
            try:
                s = SharedMemory(name=f"{key}")
                s.close()
                s.unlink()
                print(f"Successfully unlinked {key}")
            except Exception as e:
                print(e)

    def _worker_process(
        self, frames: tuple[int, int], shm_mem: dict[str, SharedMemory]
    ) -> None:
        """
        Load frames and store the data in shared memory.

        Args:
            frames (tuple): A tuple containing the start and end frame indices to process.
            shm_mem (dict): A dictionary of shared memory buffers.

        Returns:
            None
        """
        start, end = frames
        for frame in tqdm.tqdm(range(start, end)):
            frame_data = self._get_frame(frame)
            for key, value in frame_data.items():
                if key in self.data_keys:
                    shm_mem_traj = np.ndarray(
                        shape=value.shape,
                        dtype=value.dtype,
                        buffer=shm_mem[key].buf,
                        offset=(frame - self.start) * self.data2nbytes[key],
                    )
                    shm_mem_traj[:] = value[:]

    def _load_data(self) -> None:
        """
        Load data into shared memory using multiple processes.

        This method populates shared memory with data by creating multiple processes
        and assigning each process to load a specific episode of data. The data is
        stored in a shared dictionary.

        Returns:
            None
        """
        # Populate shared memory with data
        manager = mp.Manager()
        data = manager.dict()

        processes = [
            mp.Process(
                target=self._worker_process,
                args=(
                    eps,
                    self.shm_mem,
                ),
            )
            for eps in self.episodes
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        self.data = data
        print("Data loaded")


def get_episode(frame: tuple, episodes: list[tuple]) -> tuple:
    """
    Get the episode that contains the given frame.

    Args:
        frame (tuple): The frame to check.
        episodes (list[tuple]): The list of episodes.

    Returns:
        tuple: The episode that contains the frame, or None if no episode contains the frame.
    """
    for eps in episodes:
        if eps[0] <= frame[0] and frame[1] <= eps[1]:
            return tuple(eps)
