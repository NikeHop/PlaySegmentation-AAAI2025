"""Segment the data via play segmentation"""

import argparse
import os
import pickle
import random

from collections import defaultdict

import blosc
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import tqdm
import yaml

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import write_video


from play_segmentation.data.babyai.utils import (
    CL2INSTRUCTION_GOTOLOCAL,
    CL2INSTRUCTION_TYPE_GOTOLOCAL,
    state2img,
)
from play_segmentation.segmentation_models.play_segment.trainer import (
    PlaySegmentation,
)
from play_segmentation.labelling.labelling.utils import (
    create_video_with_text,
    create_babyai_video,
    load_data,
)


class LabellingPSDatasetBabyAI(Dataset):
    """
    Dataset class for labeling trajectory segments in the BabyAI environment.

    Args:
        dataset (dict): The dataset containing trajectory data.
        inst2cl (dict): A mapping of instructions to classes.

    Attributes:
        dataset (dict): The dataset containing trajectory data.
        length (int): The length of the dataset.
        inst2cl (dict): A mapping of instructions to classes.
    """

    def __init__(self, dataset, inst2cl):
        self.dataset = dataset
        self.length = len(self.dataset["images"])
        self.inst2cl = inst2cl

    def __getitem__(self, index):
        # Video
        unseg_traj = self.dataset["images"][index]
        unseg_traj = [
            blosc.unpack_array(segment)[:-1] for segment in unseg_traj[:-1]
        ] + [blosc.unpack_array(unseg_traj[-1])]
        unseg_traj = torch.from_numpy(np.concatenate(unseg_traj, axis=0)).float()

        # Collect the data necessary to form a new sample
        traj = {}
        traj["obs"] = unseg_traj

        # States
        states = self.dataset["states"][index]
        states = [blosc.unpack_array(segment)[:-1] for segment in states[:-1]] + [
            blosc.unpack_array(states[-1])
        ]
        traj["states"] = np.concatenate(states, axis=0)

        actions = [a for action in self.dataset["actions"][index] for a in action]
        traj["actions"] = actions

        directions = [
            d for direction in self.dataset["directions"][index] for d in direction
        ]
        traj["directions"] = directions

        rewards = [r for reward in self.dataset["rewards"][index] for r in reward]
        traj["rewards"] = rewards

        # Determine the stop actions
        seg_points = [len(actions) for actions in self.dataset["actions"][index]]
        stop_actions = []
        for t in seg_points:
            stop_actions += [0] * (t - 1) + [1]
        traj["stop_actions"] = stop_actions
        traj["obs_lengths"] = torch.tensor([traj["obs"].shape[0]])
        traj["seg_points"] = torch.tensor(seg_points).cumsum(-1).tolist()

        # Add classes to traj
        instructions = self.dataset["instructions"][index]
        traj["nl_instructions"] = instructions

        gt_cls = []
        for inst, actions in zip(
            self.dataset["instructions"][index], self.dataset["actions"][index]
        ):
            cl = self.inst2cl[inst]
            gt_cls += [cl] * len(actions)

        traj["gt_cls"] = gt_cls

        return traj

    def __len__(self):
        return self.length


def labelled_segmentation_ps(config: dict) -> None:
    """
    Perform labelled segmentation based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Raises:
        NotImplementedError: If the environment specified in the configuration has not been implemented.

    """
    if config["env"] == "babyai":
        labelled_segmentation_ps_babyai(config)
    elif config["env"] == "calvin":
        labelled_segmentation_ps_calvin(config)
    else:
        raise NotImplementedError(
            f"This environment {config['env_name']} has not been implemented"
        )


def labelled_segmentation_ps_calvin(config: dict) -> None:
    """
    Perform play segmentation on play trajectories from the CALVIN environment.

    Args:
        config (dict): Configuration parameters for the segmentation.

    Returns:
        None
    """

    # Sample random index for logging
    dir_index = random.randint(0, 1000)
    wandb.log({"Index": dir_index})

    # Vis Directory
    visualisation_directory = os.path.join(
        config["vis_directory"],
        "play_segmentation",
        str(dir_index),
    )
    os.makedirs(visualisation_directory, exist_ok=True)

    # Segmentation Directory
    segmentation_directory = os.path.join(
        config["segmentation_directory"], str(dir_index)
    )
    if not os.path.exists(segmentation_directory):
        os.makedirs(segmentation_directory)

    # Load Model
    model = PlaySegmentation.load_from_checkpoint(
        config["checkpoint"], map_location=config["device"]
    )

    # Load data: Episodes
    episode_directory = os.path.join(config["dataset_directory"], "training")
    with open(os.path.join(episode_directory, "ep_start_end_ids.npy"), "rb") as file:
        episodes = np.load(file)

    # Load data: full_annotation
    data_directory = os.path.join(config["dataset_directory"], "training")
    annotation_directory = os.path.join(
        config["dataset_directory"], "training", "lang_annotations"
    )

    full_annotation = np.load(
        os.path.join(annotation_directory, "auto_lang_ann.npy"), allow_pickle=True
    ).item()

    # Load data: row2task, instruction2emb, task2instructions
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

    episode = config["episode"]
    episode_segmentation = {"segmentation": [], "labels": [], "label_probs": []}
    episode_start = episodes[episode][0][0]
    episode_end = episodes[episode][0][1]
    current_timestep = episode_start
    episode_filename = f"{episode_start}_{episode_end}.pkl"
    pbar = tqdm.tqdm(total=episode_end - episode_start)
    n_visualisations = 0
    while current_timestep < episode_end:
        if episode_end - current_timestep < 64:
            break

        # Load the data
        data = load_data(
            (
                current_timestep,
                current_timestep + config["segmentation"]["segment_length"],
            ),
            data_directory,
        )

        # Perform segmentation
        optimal_segmentation, labels, label_probs = dp_segmentation(model, data, config)

        # Add segments to dataset
        start = 0
        for i in range(len(optimal_segmentation)):
            end = optimal_segmentation[i]
            prob = label_probs[i].item()
            cl = labels[i].item()
            inst = row2task[cl]
            episode_segmentation["segmentation"].append(
                (current_timestep + start, current_timestep + end)
            )
            episode_segmentation["labels"].append(cl)
            episode_segmentation["label_probs"].append(prob)

            # Visualize segmentation for debugging
            if n_visualisations < config["n_visualisations"]:
                video = data["rgb_obs"][start:end]
                write_video(
                    f"{visualisation_directory}/{inst}_{prob}.mp4", video, fps=15
                )
            start = end

        pbar.update(optimal_segmentation[-1])
        wandb.log({"n_steps": current_timestep - episode_start})
        current_timestep += optimal_segmentation[-1]

        # Save episode dataset
        with open(os.path.join(segmentation_directory, episode_filename), "wb") as file:
            pickle.dump(episode_segmentation, file)


def labelled_segmentation_ps_babyai(config: dict) -> None:
    """
    Perform labelled segmentation for the BabyAI environment using the PlaySegmentation model.

    Args:
        config (dict): Configuration parameters for the segmentation process.

    Returns:
        None
    """
    # Index for logging
    dir_index = random.randint(0, 1000)
    wandb.log({"dir_index": dir_index})

    # Create Directories
    segmentation_directory = os.path.join(
        config["segmentation_directory"], str(dir_index)
    )
    os.makedirs(segmentation_directory, exist_ok=True)

    visualisation_directory = os.path.join(
        config["vis_directory"], "play_segmentation", str(dir_index)
    )
    os.makedirs(visualisation_directory, exist_ok=True)

    # Load model
    model = PlaySegmentation.load_from_checkpoint(
        config["checkpoint"], map_location=config["device"]
    )
    model.eval()

    # Load dataset of unsegmented trajectories
    unsegmented_dataset_file = os.path.join(
        config["dataset_directory"], config["dataset_file"]
    )
    with open(unsegmented_dataset_file, "rb") as file:
        unsegmented_dataset = pickle.load(file)

    # Load dataset of segmented trajectories
    augmented_dataset_file = os.path.join(
        config["dataset_directory"], config["labelled_dataset_file"]
    )
    with open(augmented_dataset_file, "rb") as file:
        augmented_dataset = pickle.load(file)

    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
        cl2inst_type = CL2INSTRUCTION_TYPE_GOTOLOCAL
    else:
        raise NotImplementedError("This environment is not implemented")
    inst2cl = {v: k for k, v in cl2inst.items()}

    # Create dataset
    dataset = LabellingPSDatasetBabyAI(unsegmented_dataset, inst2cl)

    # Statistics
    n_labelled_segments = 0
    accuracies = []
    segment_lengths = []
    instructions = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    n_visualisations = 0

    pbar = tqdm.tqdm(total=len(dataset))
    for i, traj in enumerate(dataset):
        # Perform Dynamic Programming (DP) segmentation
        optimal_segmentation, labels, _ = dp_segmentation(model, traj, config)

        # Update statistics
        for seg_point in optimal_segmentation:
            if seg_point in traj["seg_points"]:
                true_positives += 1
            else:
                false_positives += 1

        for seg_point in traj["seg_points"]:
            if seg_point not in optimal_segmentation:
                false_negatives += 1

        # Process segmentation to samples
        new_samples, accs = process_segmentation_to_samples_babyai(
            optimal_segmentation,
            labels,
            traj,
            cl2inst,
            cl2inst_type,
            i,
            visualisation_directory,
            config,
        )
        accuracies += accs

        # Add samples
        for i, new_sample in enumerate(new_samples):
            (
                video,
                state,
                instruction,
                instruction_type,
                action,
                direction,
                reward,
            ) = new_sample
            augmented_dataset["images"].append(video)
            augmented_dataset["states"].append(state)
            augmented_dataset["actions"].append(action)
            augmented_dataset["instructions"].append(instruction)
            augmented_dataset["rewards"].append(reward)
            augmented_dataset["directions"].append(direction)
            augmented_dataset["instruction_types"].append(instruction_type)
            n_labelled_segments += 1

            # Add logging-info
            segment_lengths.append(len(action) + 1)
            instructions.append(instruction)

            if n_visualisations < config["n_visualisations"]:
                create_babyai_video(
                    blosc.unpack_array(new_samples[i][1]),
                    new_samples[i][2],
                    visualisation_directory,
                    n_labelled_segments,
                )
                n_visualisations += 1

        pbar.update(1)

        if i > 0 and config["debug"]:
            return

    # Log accuracy
    acc = np.mean(accuracies)
    wandb.log({"accuracy": acc})
    wandb.log({"Avg. Length": np.array(segment_lengths).mean()})
    wandb.log({"Precision": true_positives / (true_positives + false_positives)})
    wandb.log({"Recall": true_positives / (true_positives + false_negatives)})

    # Plot histogram of segment lengths
    segment_length_types = defaultdict(int)
    for segment_length in segment_lengths:
        segment_length_types[segment_length] += 1

    plt.bar(segment_length_types.keys(), segment_length_types.values())
    plt.xlabel("Segment Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"ps_segment_length.png")
    plt.savefig(f"ps_segment_length.pdf")

    # Plot histogram of instructions
    instruction_types = defaultdict(int)
    for instruction in instructions:
        instruction_types[instruction] += 1

    plt.clf()
    plt.bar(instruction_types.keys(), instruction_types.values())
    plt.xticks(rotation=45)
    plt.xlabel("Instruction Types")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"ps_instruction_types.png")
    plt.savefig(f"ps_instruction_types.pdf")

    # Save the augmented dataset
    filename = (
        f"{config['labelled_dataset_file'].split('.')[0]}_ps_{n_labelled_segments}.pkl"
    )
    bc_dataset_file = os.path.join(config["dataset_directory"], filename)
    with open(bc_dataset_file, "wb") as file:
        pickle.dump(augmented_dataset, file)


def process_segmentation_to_samples_babyai(
    optimal_segmentation: list[int],
    labels: torch.Tensor,
    data: dict[str, torch.Tensor],
    cl2inst: dict[int, str],
    cl2inst_type: dict[int, str],
    trajectory_id: str,
    vis_directory: str,
    config: dict[str, any],
) -> tuple:
    """
    Process the segmentation and generate samples for the BabyAI dataset.

    Args:
        optimal_segmentation (list): List of indices representing the optimal segmentation.
        labels (torch.Tensor): Tensor of labels for each segment.
        data (dict): Dictionary containing the data for the trajectory.
        cl2inst (dict): Mapping of class labels to instructions.
        cl2inst_type (dict): Mapping of class labels to instruction types.
        trajectory_id (str): ID of the trajectory.
        vis_directory (str): Directory to save visualizations.
        config (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing the new samples and accuracies.
            - new_samples (list): List of new samples.
            - accuracies (list): List of accuracies for each segment.
    """
    # Create new samples
    new_samples = []
    instructions = []
    accuracies = []
    start = 0
    for i in range(len(optimal_segmentation)):
        end = optimal_segmentation[i]
        video = blosc.pack_array(data["obs"][start : end + 1].cpu().numpy())
        states = blosc.pack_array(data["states"][start : end + 1])
        action = data["actions"][start:end]
        direction = data["directions"][start:end]
        reward = data["rewards"][start:end]
        instruction = cl2inst[labels[i].item()]
        instructions.append(instruction)
        instruction_type = cl2inst_type[labels[i].item()]
        accuracies += [labels[i].item() == gt_cl for gt_cl in data["gt_cls"][start:end]]

        new_sample = [
            video,
            states,
            instruction,
            instruction_type,
            action,
            direction,
            reward,
        ]
        new_samples.append(new_sample)
        start = end

    # Create visualization
    if config["visualize"] and config["debug"]:
        video = []

        for i in range(len(data["states"])):
            img = state2img(data["states"][i])
            video.append(img)

        output_path = os.path.join(vis_directory, f"full_seg_{trajectory_id}.mp4")
        video_array = np.stack(video, axis=0)
        descriptions = {
            cutoff + i: inst
            for i, (cutoff, inst) in enumerate(zip(optimal_segmentation, instructions))
        }
        create_video_with_text(
            video_array, descriptions, output_path, fps=2, added_height=50
        )

    return new_samples, accuracies


def dp_segmentation(model: torch.nn.Module, data: dict, config: dict) -> tuple:
    """
    Perform dynamic programming segmentation on the given trajectory segment.

    Args:
        model: The segmentation model.
        data: The input data for segmentation.
        config: The configuration parameters for segmentation.

    Returns:
        A tuple containing the optimal segmentation, labels, and label probabilities.
    """

    w_min = config["segmentation"]["w_min"]
    w_max = config["segmentation"]["w_max"]

    # Preprocessing
    stop_log_probs, cls_log_probs = get_probs_ps(model, data, config)
    segmentation = initialize_segmentation(
        stop_log_probs, cls_log_probs, (w_min, w_max)
    )
    T = data["rgb_obs"].shape[0]
    max_n_intervals = T if w_min == 1 else (T // w_min) + 1

    for i in range(2, max_n_intervals):
        for k in range(i * w_min, min(i * w_max, T)):
            starts = list(range(max(k - w_max, (i - 1) * w_min), k - w_min + 1))
            ends = [k] * len(starts)

            (scores, labels, label_probs) = calculate_score(
                stop_log_probs, cls_log_probs, starts, ends
            )

            for n, l in enumerate(
                range(
                    max(k - w_max, (i - 1) * w_min),
                    min(k - w_min + 1, w_max * (i - 1)),
                )
            ):
                score = scores[n]
                label = labels[n]
                label_prob = label_probs[n]

                if (i, k) not in segmentation["p"]:
                    segmentation["p"][(i, k)] = segmentation["p"][(i - 1, l)] + score
                    segmentation["s"][(i, k)] = segmentation["s"][(i - 1, l)] + [k]
                    segmentation["l"][(i, k)] = segmentation["l"][(i - 1, l)] + [label]
                    segmentation["lp"][(i, k)] = segmentation["lp"][(i - 1, l)] + [
                        label_prob
                    ]

                if segmentation["p"][(i, k)] < segmentation["p"][(i - 1, l)] + score:
                    segmentation["p"][(i, k)] = segmentation["p"][(i - 1, l)] + score
                    segmentation["s"][(i, k)] = segmentation["s"][(i - 1, l)] + [k]
                    segmentation["l"][(i, k)] = segmentation["l"][(i - 1, l)] + [label]
                    segmentation["lp"][(i, k)] = segmentation["lp"][(i - 1, l)] + [
                        label_prob
                    ]

    optimal_segmentation, labels, label_probs = filter_optimal_segmentation(
        segmentation, max_n_intervals, T, stop_log_probs, config
    )

    return optimal_segmentation, labels, label_probs


def get_probs_ps(
    model: torch.nn.Module, data: dict, config: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the log-probability matrices for trajectory segmentation.

    Args:
        model (torch.nn.Module): The play segmentation model.
        data (dict): The input data containing observations.
        config (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing the log-probability matrices for stop probabilities and class probabilities.
    """
    w_max = config["segmentation"]["w_max"]
    with torch.no_grad():
        # Prepare log-probability matrices
        T = data["rgb_obs"].shape[0]
        B = config["num_classes"]

        # Entry [i,j] represents the log-probability of o_i_j being a segment
        stop_log_probs = -float("inf") * torch.ones(T - 1, T)
        # Entry [i,j] represents the log-probability distribution over instructions for segment o_i_j
        cls_log_probs = -float("inf") * torch.ones(T - 1, T, B)

        # Iterate over all possible starting states
        for i in range(T - 1):
            # Consider all possible segments starting in i constrained by the max segment length
            obss = pad_sequence(
                [data["rgb_obs"][i:j] for j in range(i + 2, min(i + w_max, T + 1))],
                batch_first=True,
            )
            obs_length = torch.arange(2, min(w_max, T + 1 - i))

            # Iterate over data in batches to avoid memory issues
            steps = 0
            while steps < obss.shape[0]:
                next_step = min(steps + config["batch_size"], obss.shape[0])

                batch_obss = obss[steps:next_step]
                batch_obs_length = obs_length[steps:next_step]
                batch = {
                    "img_obs": batch_obss.to(config["device"]),
                    "obs_length": batch_obs_length.to(config["device"]),
                }

                stop_log_prob_ind, cls_log_prob_ind = model.get_log_probs(batch)

                for k, step in enumerate(range(steps, next_step)):
                    stop_log_probs[i, i + 1 + step] = stop_log_prob_ind[
                        k, batch_obs_length[k] - 1
                    ]
                    cls_log_probs[i, i + 1 + step] = cls_log_prob_ind[
                        k, batch_obs_length[k] - 1
                    ]

                steps = next_step

    return stop_log_probs, cls_log_probs


def initialize_segmentation(stop_log_probs, cls_log_probs, w):
    """
    Initializes the segmentation dictionary.

    Parameters:
    - stop_log_probs (torch.Tensor): Array of stop log probabilities.
    - cls_log_probs (torch.Tensor): Array of class log probabilities.
    - w (tuple): Tuple of minimum and maximum window sizes.

    Returns:
    - segmentation (dict): Segmentation dictionary with initialized values.

    """
    w_min, w_max = w

    segmentation = {
        "p": defaultdict(float),
        "s": defaultdict(list),
        "l": defaultdict(list),
        "lp": defaultdict(list),
    }

    T = stop_log_probs.shape[0]

    ends = range(w_min, min(T, w_max))
    starts = [0] * len(ends)

    scores, labels, label_probs = calculate_score(
        stop_log_probs, cls_log_probs, starts, ends
    )

    for i in range(w_min, min(T, w_max)):
        segmentation["p"][(1, i)] = scores[i - w_min]
        segmentation["s"][(1, i)] = [i]
        segmentation["l"][(1, i)] = [labels[i - w_min]]
        segmentation["lp"][(1, i)] = [label_probs[i - w_min]]

    return segmentation


def calculate_score(
    stop_log_probs: torch.Tensor,
    cls_log_probs: torch.Tensor,
    starts: list[int],
    ends: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Calculate the scores, labels, and label probabilities for a given set of intervals,

    Args:
        stop_log_probs (torch.Tensor): The stop log probabilities.
        cls_log_probs (torch.Tensor): The class log probabilities.
        starts (List[int]): The start indices of the intervals.
        ends (List[int]): The end indices of the intervals.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: A tuple containing the maximum scores,
        labels, and label probabilities for each segment.
    """
    max_scores = []
    labels = []
    label_probs = []
    stop_log_probs_neg = torch.log(1 - torch.exp(stop_log_probs))

    for start, end in zip(starts, ends):
        stop_scores = torch.cat(
            [
                stop_log_probs_neg[
                    start,
                    start + 1 : end,
                ],
                stop_log_probs[start, end].unsqueeze(0),
            ],
            dim=0,
        ).sum(dim=0)

        action_scores = cls_log_probs[start, end]
        score = stop_scores

        label = action_scores.argmax()
        label_prob = action_scores.max()
        max_scores.append(score)
        labels.append(label)
        label_probs.append(torch.exp(label_prob))

    return max_scores, labels, label_probs


def filter_optimal_segmentation(
    segmentation: dict,
    n_intervals: int,
    n_steps: int,
    stop_log_prob: torch.Tensor,
    config: dict,
) -> tuple:
    """
    Filters the optimal segmentation based on the given parameters.

    Args:
        segmentation (dict): The segmentation dictionary containing the probabilities and labels.
        n_intervals (int): The number of intervals.
        n_steps (int): The number of steps.
        stop_log_prob (torch.Tensor): The stop log probabilities.
        config (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing the optimal segmentation, labels, and label probabilities.
    """
    max_score = -float("inf")
    optimal_segmentation = []
    labels = []

    for i in range(n_intervals):
        if (i, n_steps - 1) in segmentation["p"]:
            if config["debug"]:
                print(
                    "#" * 15,
                    "Optimal Segmetation for each number of possible intervals",
                    "#" * 15,
                )
                log_segmentation(i, n_steps, segmentation)

            if segmentation["p"][(i, n_steps - 1)] > max_score:
                optimal_segmentation = segmentation["s"][(i, n_steps - 1)]
                labels = segmentation["l"][(i, n_steps - 1)]
                max_score = segmentation["p"][(i, n_steps - 1)]
                label_probs = segmentation["lp"][(i, n_steps - 1)]

    if config["check_last_valid_segment"]:
        if len(optimal_segmentation) >= 2:
            start = optimal_segmentation[-2]
            end = optimal_segmentation[-1]
        else:
            start = 0
            end = optimal_segmentation[-1]

        if torch.exp(stop_log_prob[start, end]) < 0.5:
            optimal_segmentation = optimal_segmentation[:-1]
            labels = labels[:-1]
            label_probs = label_probs[:-1]

    return optimal_segmentation, labels, label_probs


def log_segmentation(n_intervals: int, n_timesteps: int, segmentation: dict) -> None:
    """
    Logs the segmentation information.

    Args:
        n_intervals (int): Number of intervals.
        n_timesteps (int): Number of timesteps.
        segmentation (dict): Segmentation data.

    Prints the following information for the optimal segmentation of the interval for each number of intervals:
    - Number of Timesteps
    - Score
    - Segmentation Points
    - Labels
    - Label Probabilities
    """
    print(f"Number of Intervals {n_intervals}")
    print(f"Number of Timesteps {n_timesteps}")
    print(f"Score {segmentation['p'][(n_intervals, n_timesteps - 1)]}")
    print(f"Segmentation Points {segmentation['s'][(n_intervals, n_timesteps - 1)]}")
    print(
        f"Labels {[elem.item() for elem in segmentation['l'][(n_intervals, n_timesteps - 1)]]}"
    )
    print(
        f"Label Probabilities {['{:.2f}'.format(elem.item()) for elem in segmentation['lp'][(n_intervals, n_timesteps - 1)]]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/label_stop_policy.yaml",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="Path to checkpoint"
    )
    parser.add_argument(
        "--calvin_episode", default=None, type=int, help="Episode of Calvin to segment"
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug mode"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Update config with CLI arguments
    if config["env"] == "calvin":
        assert (
            args.calvin_episode is not None
        ), "Please provide an episode to segment for the CALVIN environment."

    if args.calvin_episode is not None:
        config["episode"] = [args.calvin_episode]

    if args.checkpoint is not None:
        config["checkpoint"] = args.checkpoint

    if args.debug:
        config["debug"] = True

    # Init wandb
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"],
        dir="../results",
    )

    labelled_segmentation_ps(config)
