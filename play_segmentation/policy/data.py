import os
import random
import pickle

import blosc
import numpy as np
import torch

from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5EncoderModel

from play_segmentation.data.babyai.utils import (
    ALL_INSTRUCTIONS_GOTOLOCAL,
)
from play_segmentation.utils.data import BaseDataset


class BabyAIDataset(Dataset):
    """
    Dataset class for BabyAI dataset.

    Args:
        data (dict): The dataset containing states, actions, and instructions.
        inst2embeddings (dict): A dictionary mapping instructions to embeddings.
        use_state (bool): Flag indicating whether to use states or images.

    Attributes:
        trajs (list): List of tuples containing observation, action, and instruction for each trajectory.
        inst2embedding (dict): A dictionary mapping instructions to embeddings.
    """

    def __init__(self, data: dict, inst2embeddings: dict, use_state: bool):
        self.trajs = []
        self.inst2embedding = inst2embeddings

        if use_state:
            obss = data["states"]
        else:
            obss = data["images"]

        for obs, action, instr in zip(obss, data["actions"], data["instructions"]):
            self.trajs.append((obs, action, instr))

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Get an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple[np.ndarray, int, np.ndarray]: A tuple containing the observation, action, and instruction.
        """
        traj = self.trajs[index]
        obs = blosc.unpack_array(traj[0])
        action = traj[1]
        instruction = self.encode_instruction(traj[2])

        if len(action) == 0:
            return None, None, None

        timestep = random.randint(0, len(action) - 1)

        obs = obs[timestep]
        action = action[timestep]

        return obs, action, instruction

    def __len__(self) -> int:
        """
        Returns the length of the data object.

        Returns:
            int: The number of trajectories in the data object.
        """
        return len(self.trajs)

    def encode_instruction(self, instruction: str) -> torch.Tensor:
        """
        Encodes the given instruction into an embedding.

        Args:
            instruction (str): The instruction to be encoded.

        Returns:
            embedding: The embedding of the instruction.
        """
        embedding = self.inst2embedding[instruction]
        return embedding


class CalvinDataset(BaseDataset):
    """
    Dataset class for CalvinDataset.

    Args:
        split (str): The split of the dataset (e.g., "train", "validation").
        dataset_directory (str): The directory where the dataset is located.
        episodes (list): List of episodes in the dataset.
        split_episodes (list): List of episodes for the current split.
        frames2row (dict): Mapping of frames to rows in the dataset.
        frames2split (dict): Mapping of frames to splits in the dataset.
        instruction_embeddings (torch.Tensor): Tensor of instruction embeddings.
        ratio (float): The ratio of img to language goal embeddings.
        debug (bool): Flag indicating whether to enable debug mode.
    """

    def __init__(
        self,
        split,
        dataset_directory,
        episodes,
        split_episodes,
        frames2row,
        frames2split,
        instruction_embeddings,
        ratio,
    ):
        self.data_keys = [
            "rgb_static",
            "rgb_gripper",
            "rel_actions",
        ]

        super().__init__(
            split,
            dataset_directory,
            episodes,
            split_episodes,
            frames2row,
            frames2split,
        )

        self.ratio = ratio
        self.instruction_embeddings = instruction_embeddings

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        if self.split == "validation":
            return len(self.frames2row)
        else:
            return self.dataset_size

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, torch.Tensor, np.ndarray, torch.Tensor, tuple]:
        """
        Retrieves an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the following elements:
                - d["rgb_static"] (np.ndarray): The static RGB data.
                - actions (torch.Tensor): The relative actions.
                - d["rgb_gripper"] (np.ndarray): The gripper RGB data.
                - inst (torch.Tensor): The instruction embedding.
                - (start_frame, end_frame) (tuple): The start and end frames.
        """
        if random.random() > self.ratio:
            # Sample episode
            start_frame, end_frame = random.choice(self.episodes)
            # Sample start time
            start_time = random.randint(
                0, end_frame - start_frame - self.window_size_high
            )
            # Sample window size
            window_size = random.randint(self.window_size_low, self.window_size_high)

            start_frame = start_frame + start_time
            end_frame = start_frame + window_size

            inst = torch.zeros(1, 384)

        else:
            start_frame, end_frame = random.choice(list(self.frames2row.keys()))

            inst = self.instruction_embeddings[
                self.frames2row[(start_frame, end_frame)]
            ].reshape(1, -1)

        d = {}
        for key in self.data_keys:
            shape = (end_frame - start_frame, *self.data2shape[key])
            d[key] = np.ndarray(
                shape=shape,
                dtype=self.data2dtype[key],
                buffer=self.shm_mem[key].buf,
                offset=(start_frame - self.start) * self.data2nbytes[key],
            )

        actions = torch.from_numpy(d["rel_actions"])

        return (
            d["rgb_static"],
            actions,
            d["rgb_gripper"],
            inst,
            (start_frame, end_frame),
        )


def get_data(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Get data based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing the dataset information.

    Returns:
        data: Data based on the specified dataset.

    Raises:
        ValueError: If the specified dataset is not supported.
    """
    if config["dataset"] == "mcil":
        return get_data_mcil(config)
    elif config["dataset"] == "babyai":
        return get_data_babyai(config)
    else:
        raise ValueError(f"Dataset {config['dataset']} not supported")


def get_data_babyai(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Get the training and validation data loaders for the BabyAI dataset.

    Args:
        config (dict): Configuration parameters for the dataset.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    with open(
        os.path.join(config["dataset_directory"], config["dataset_file"]), "rb"
    ) as file:
        train_data = pickle.load(file)

    with open(
        os.path.join(
            config["dataset_directory"], "babyai_go_to_10000_single_validation.pkl"
        ),
        "rb",
    ) as file:
        val_data = pickle.load(file)

    inst2embeddings = get_embeddings(config)
    use_state = config["use_state"]

    train_dataset = BabyAIDataset(train_data, inst2embeddings, use_state)
    val_dataset = BabyAIDataset(val_data, inst2embeddings, use_state)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=collate_babyai,
        num_workers=config["num_workers"],
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=collate_babyai,
        num_workers=config["num_workers"],
    )

    return train_dataloader, val_dataloader


def get_embeddings(config: dict) -> dict:
    """
    Get or generate instruction embeddings based on the given configuration.

    Args:
        config (dict): The configuration containing the environment name and instruction embeddings type.

    Returns:
        dict: A dictionary mapping instructions to their corresponding embeddings.
    """
    if config["env_name"] == "go_to":
        instructions = ALL_INSTRUCTIONS_GOTOLOCAL
    else:
        raise NotImplementedError("Environment not supported")

    # Check whether embeddings are already generated
    ins_embedding_type = config["instruction_embeddings"]["type"]
    embeddings_file = os.path.join(
        config["dataset_directory"],
        f"embeddings_{config['env_name']}_{ins_embedding_type}.pkl",
    )

    if os.path.exists(embeddings_file):
        with open(embeddings_file, "rb") as file:
            inst2embeddings = torch.load(file)
    else:
        inst2embeddings = generate_embeddings(instructions, config)
        # Save embeddings
        with open(embeddings_file, "wb") as file:
            torch.save(inst2embeddings, file)

    return inst2embeddings


def generate_embeddings(
    instructions: list[str], config: dict
) -> dict[str, torch.Tensor]:
    """
    Generate embeddings for a list of instructions based on the given configuration.

    Args:
        instructions (list): A list of instructions.
        config (dict): A dictionary containing the configuration for generating embeddings.

    Returns:
        dict: A dictionary containing the generated embeddings for each instruction.
    """
    if config["instruction_embeddings"]["type"] == "t5":
        embeddings_dict = generate_t5_embeddings(instructions, config)

    elif config["instruction_embeddings"]["type"] == "hot_one":
        embeddings_dict = generate_hot_one_embeddings(instructions)

    else:
        raise NotImplementedError("Embedding type not supported")

    return embeddings_dict


def generate_hot_one_embeddings(instructions: list[str]) -> dict[str, torch.Tensor]:
    """
    Generate one-hot embeddings for a list of instructions.

    Args:
        instructions (list): A list of instructions.

    Returns:
        dict: A dictionary mapping each instruction to its corresponding one-hot embedding.
    """
    emb_dict = {}
    n_inst = len(instructions)
    instruction_embeddings = torch.eye(n_inst)

    for inst, emb in zip(instructions, instruction_embeddings):
        emb_dict[inst] = emb

    return emb_dict


def generate_t5_embeddings(
    instructions: list[str], config: dict[str, any]
) -> dict[str, torch.Tensor]:
    """
    Generates T5 embeddings for a list of instructions.

    Args:
        instructions (list): List of instructions to generate embeddings for.
        config (dict): Configuration dictionary containing model information.

    Returns:
        dict: A dictionary mapping each instruction to its corresponding T5 embedding.
    """
    instructions = instructions
    embeddings_dict = {}

    tokenizer = AutoTokenizer.from_pretrained(config["instruction_embeddings"]["model"])

    encoder_model = T5EncoderModel.from_pretrained(
        config["instruction_embeddings"]["model"]
    )

    inputs = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        model_output = encoder_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    embeddings = model_output.last_hidden_state.mean(dim=1)

    for elem, instruction in zip(embeddings, instructions):
        embeddings_dict[instruction] = elem

    return embeddings_dict


def get_data_mcil(config: dict) -> tuple:
    """
    Get the training and validation datasets for the MCIL task.

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
    splits = ["training", "validation"]

    for split in splits:
        split_episodes = split2episodes[split]
        episode_directory = os.path.join(config["dataset_directory"], split)

        if split == "training":
            filename = f'{config["episode_filename"]}'
        else:
            filename = "auto_lang_ann.npy"

        print(f"Loading episodes from {filename}")

        with open(
            os.path.join(episode_directory, f"lang_annotations/{filename}"),
            "rb",
        ) as file:
            lang = np.load(file, allow_pickle=True).item()

            if config["task_embeddings"]:
                instruction_embeddings = (
                    torch.from_numpy(lang["language"]["task_emb"]).float().squeeze(1)
                )
                frame2row = lang["language"]["task_frame2row"]
                frame2row = {
                    traj: frame2row[traj] for i, traj in enumerate(lang["info"]["indx"])
                }
            else:
                instruction_embeddings = (
                    torch.from_numpy(lang["language"]["emb"]).float().squeeze(1)
                )
                frame2row = {traj: i for i, traj in enumerate(lang["info"]["indx"])}

        if split == "validation":
            # Only sample from the labelled data at evaluation time
            ratio = 1
        else:
            # Sample with equal prob from labelled and unlabelled data
            ratio = 0.5

        dataset = CalvinDataset(
            split,
            config["dataset_directory"],
            episodes,
            split_episodes,
            frame2row,
            frames2split,
            instruction_embeddings,
            ratio,
        )

        if split == "training":
            shuffle = True
        else:
            shuffle = False

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=config["batch_size"],
            collate_fn=collate_calvin,
            num_workers=config["num_workers"],
        )

        datasets[split] = dataloader

    return datasets["training"], datasets["validation"]


def collate_babyai(data: list) -> dict:
    """
    Collates a list of data samples into a batch.

    Args:
        data (list): A list of tuples, where each tuple contains an observation, an action, and an instruction.

    Returns:
        dict: A dictionary containing the collated batch, with the following keys:
            - "actions": A tensor of actions.
            - "obss": A tensor of observations.
            - "instructions": A tensor of instructions.
    """
    img_obs = []
    instructions = []
    actions = []

    for obs, action, instruction in data:
        if obs is None:
            continue
        img_obs.append(torch.tensor(obs).float())
        actions.append(action)
        instructions.append(instruction)

    obss = torch.stack(img_obs, dim=0)
    obss = rearrange(obss, "b h w c -> b c h w")
    actions = torch.tensor(actions, dtype=torch.long)
    instructions = torch.stack(instructions, dim=0)

    batch = {
        "actions": actions,
        "obss": obss,
        "instructions": instructions,
    }

    return batch


def collate_calvin(data: list) -> dict:
    """
    Collates a batch of data samples into a dictionary.

    Args:
        data (list): List of data samples, where each sample is a tuple containing:
            - Image observations
            - Actions
            - Gripper observations
            - Instructions
            - Index

    Returns:
        dict: A dictionary containing the collated batch of data with the following keys:
            - "img_obs": Padded image observations
            - "gripper_obs": Padded gripper observations
            - "obs_length": Lengths of image observations
            - "instructions": Padded instructions
            - "inst_length": Lengths of instructions
            - "actions": Padded actions
            - "mask": Mask indicating non-zero instructions
            - "idx": List of indices
    """
    img_obs = []
    gripper_obs = []
    obs_length = []
    inst = []
    inst_length = []
    actions = []
    idx = []

    for i, sample in enumerate(data):
        img_obs.append(torch.tensor(np.array(sample[0]), dtype=torch.float))
        obs_length.append(img_obs[-1].shape[0])
        actions.append(sample[1])
        gripper_obs.append(torch.tensor(np.array(sample[2]), dtype=torch.float))
        inst.append(sample[3])
        idx.append(sample[4])
        inst_length.append(inst[-1].shape[0])

    obs = pad_sequence(img_obs, batch_first=True)
    gripper_obs = pad_sequence(gripper_obs, batch_first=True)
    obs_length = torch.tensor(obs_length, dtype=torch.long)
    inst = pad_sequence(inst, batch_first=True)
    inst_length = torch.tensor(inst_length, dtype=torch.long)
    actions = pad_sequence(actions, batch_first=True)
    mask = ~torch.all(inst == 0, dim=-1).squeeze()

    batch = {
        "img_obs": obs,
        "gripper_obs": gripper_obs,
        "obs_length": obs_length,
        "instructions": inst,
        "inst_length": inst_length,
        "actions": actions,
        "mask": mask,
        "idx": idx,
    }

    return batch
