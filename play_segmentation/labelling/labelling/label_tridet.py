""" """

import argparse
import os
import pickle
import random

from collections import defaultdict

import blosc
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import wandb
import yaml

from torchvision.io import write_video
from torch.utils.data import Dataset

from play_segmentation.labelling.labelling.utils import create_babyai_video, load_data
from play_segmentation.segmentation_models.tridet.trainer import TriDet
from play_segmentation.data.babyai.utils import (
    CL2INSTRUCTION_GOTOLOCAL,
    CL2INSTRUCTION_TYPE_GOTOLOCAL,
)


class LabellingDatasetBabyAITridet(Dataset):
    """
    Dataset class for labeling in the BabyAI environment using the TriDet segmentation method.

    Args:
        dataset (dict): The dataset containing the images, states, actions, rewards, directions, and instructions.
        tridet_segment_size (int): The size of the TriDet segment.
        inst2cl (dict): A mapping from instruction to class.

    Attributes:
        dataset (dict): The dataset containing the images, states, actions, rewards, directions, and instructions.
        length (int): The length of the dataset.
        tridet_segment_size (int): The size of the TriDet segment.
        inst2cl (dict): A mapping from instruction to class.
    """

    def __init__(self, dataset: dict, tridet_segment_size: int, inst2cl: dict) -> None:
        self.dataset = dataset
        self.length = len(self.dataset["images"])
        self.tridet_segment_size = tridet_segment_size
        self.inst2cl = inst2cl

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.length

    def __len__(self) -> int:
        return self.length

    def sample_segment(
        self,
    ) -> tuple[torch.Tensor, np.ndarray, list[int], list[int], list[int], list[int]]:
        """
        Samples a segment from the dataset.

        Returns:
            A tuple containing the following elements:
            - obs: A torch.Tensor representing the concatenated segments of the video.
            - states: A numpy.ndarray representing the concatenated segments of the states.
            - actions: A list of integers representing the actions.
            - rewards: A list of integers representing the rewards.
            - directions: A list of integers representing the directions.
            - gt_cls: A list of integers representing the ground truth classes.
        """
        index = random.randint(0, self.length - 1)

        ## Get the data
        # Video
        unseg_traj = self.dataset["images"][index]
        segments = [blosc.unpack_array(elem)[:-1] for elem in unseg_traj[:-1]] + [
            blosc.unpack_array(unseg_traj[-1])
        ]
        obs_seq = torch.from_numpy(np.concatenate(segments, axis=0))

        # States, Actions, Rewards, Directions
        states = self.dataset["states"][index]
        state_segments = [blosc.unpack_array(elem)[:-1] for elem in states[:-1]] + [
            blosc.unpack_array(states[-1])
        ]
        state_seq = np.concatenate(state_segments, axis=0)

        actions = [a for action in self.dataset["actions"][index] for a in action]
        rewards = [r for reward in self.dataset["rewards"][index] for r in reward]
        directions = [
            d for direction in self.dataset["directions"][index] for d in direction
        ]

        gt_cls = []
        for actions, inst in zip(
            self.dataset["actions"][index], self.dataset["instructions"][index]
        ):
            gt_cls += [self.inst2cl[inst]] * len(actions)

        # Create a window of the data
        if obs_seq.shape[0] - 1 < self.tridet_segment_size:
            return None

        start_time = random.randint(0, obs_seq.shape[0] - self.tridet_segment_size)
        obs = obs_seq[start_time : start_time + self.tridet_segment_size]
        states = state_seq[start_time : start_time + self.tridet_segment_size]
        actions = actions[start_time : start_time + self.tridet_segment_size]
        rewards = rewards[start_time : start_time + self.tridet_segment_size]
        directions = directions[start_time : start_time + self.tridet_segment_size]
        gt_cls = gt_cls[start_time : start_time + self.tridet_segment_size]

        return obs, states, actions, rewards, directions, gt_cls


def label_tridet(config: dict) -> None:
    """
    Labels trajectories using the TriDet method based on the given configuration.

    Args:
        config (dict): The configuration dictionary containing the environment information.

    Raises:
        NotImplementedError: If the specified environment is not implemented.
    """
    if config["env"] == "babyai":
        label_tridet_babyai(config)
    elif config["env"] == "calvin":
        label_tridet_calvin(config)
    else:
        raise NotImplementedError(f"Environment {config['env']} not implemented")


def label_tridet_calvin(config: dict) -> None:
    """
    Labels the frames using the TriDet model based on the given configuration.

    Args:
        config (dict): The configuration dictionary containing the necessary parameters.

    Returns:
        None
    """
    # Sample run index
    index = random.randint(0, 100000)
    wandb.log({"Index": index})

    # Visualisation Directory
    visualisation_directory = os.path.join(
        config["vis_directory"], "tridet", str(index)
    )
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
            "task_emb": [],
            "task_frame2row": [],
            "clip_frame2row": [],
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

    # Get episodes
    frame_directory = os.path.join(config["dataset_directory"], "training")
    episodes = np.load(os.path.join(frame_directory, "ep_start_end_ids.npy"))
    n_episodes = episodes.shape[0]

    # Get TriDet Model
    tridet_model = TriDet.load_from_checkpoint(
        config["checkpoint"], map_location=config["device"]
    )
    tridet_model.model.test_pre_nms_topk = 1
    tridet_model.model.eval()

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
        done = False
        if frame in partial_annotation["info"]["indx"]:
            # Copy the stuff
            gt_annotation["info"]["indx"].append(frame)
            gt_annotation["language"]["ann"].append(ann)
            gt_annotation["language"]["emb"].append(emb)
            gt_annotation["language"]["task"].append(task)
            pbar.update(1)
        else:
            while not done:
                # Sample episode
                eps = random.randint(0, n_episodes - 1)
                start, end = episodes[eps]

                # Sample start point
                start = random.randint(start, end - config["max_window_size"] - 1)

                # Load data
                sample = load_data(
                    (start, start + config["max_window_size"] - 1), frame_directory
                )
                if not sample:
                    continue

                # Crop segment
                # Prepare tridet sample
                videos = sample["rgb_obs"].to(config["device"]).unsqueeze(0)
                obs_lengths = (
                    torch.tensor([videos.shape[1]]).float().to(config["device"])
                )
                result = tridet_model.model(videos, obs_lengths, None, None)[0]

                segment_start, segment_end = int(result["segments"][-2][0]), int(
                    result["segments"][-2][1]
                )
                label = result["labels"][-2]
                score = result["scores"][-2]
                print(segment_start, segment_end, label, score)

                if segment_start < 0 or segment_end < 0:
                    continue

                if score > config["threshold"]:
                    end = start + segment_end
                    start = start + segment_start
                    frame = (start, end)

                    task = row2task[label.item()]
                    ann = random.sample(task2instructions[task], k=1)[0]
                    emb = instruction2emb[ann]

                    gt_annotation["info"]["indx"].append(frame)
                    gt_annotation["language"]["ann"].append(ann)
                    gt_annotation["language"]["emb"].append(emb)
                    gt_annotation["language"]["task"].append(task)

                    # Update statistics
                    statistics["accepted"] += 1
                    pbar.update(1)
                    done = True

                    # Visualize
                    if n_visualisations < config["n_visualisations"]:
                        video = sample["rgb_obs"]
                        video_path = os.path.join(
                            visualisation_directory, f"{n_visualisations}_{ann}.mp4"
                        )
                        write_video(video_path, video, fps=15)
                        n_visualisations += 1

                # Update statistics
                statistics["total"] += 1

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
    with open(
        os.path.join(
            annotation_directory,
            f"auto_lang_ann_tridet_{config['threshold']}_{config['percent']}_statistics.pkl",
        ),
        "wb+",
    ) as file:
        pickle.dump(statistics, file)

    # Save random annototation
    np.save(
        os.path.join(
            annotation_directory,
            f"auto_lang_ann_tridet_{config['threshold']}_{config['percent']}.npy",
        ),
        gt_annotation,
        allow_pickle=True,
    )


def label_tridet_babyai(config: dict) -> None:
    """
    Labels the unsegmented dataset using the TriDet model.

    Args:
        config (dict): Configuration parameters for the labeling process.

    Returns:
        None
    """
    index = random.randint(0, 1000)

    wandb.log({"id": index})

    # Create directory for visualizations
    vis_directory = os.path.join(
        config["visualizations_directory"], "tridet", str(index)
    )
    if not os.path.exists(vis_directory):
        os.makedirs(vis_directory)

    # Load unsegmented dataset
    unsegmented_datafile = os.path.join(
        config["dataset_directory"], config["dataset_file"]
    )
    with open(unsegmented_datafile, "rb") as file:
        dataset = pickle.load(file)

    labelled_datafile = os.path.join(
        config["dataset_directory"], config["labelled_dataset_file"]
    )

    with open(labelled_datafile, "rb") as file:
        labelled_dataset = pickle.load(file)

    if config["env_name"] == "go_to":
        cl2inst = CL2INSTRUCTION_GOTOLOCAL
        cl2inst_type = CL2INSTRUCTION_TYPE_GOTOLOCAL
    else:
        raise NotImplementedError(f"Environment {config['env']} not implemented")

    # Create segment dataset
    inst2cl = {inst: cl for cl, inst in cl2inst.items()}
    dataset = LabellingDatasetBabyAITridet(
        dataset, config["tridet_segment_size"], inst2cl
    )

    # Load model
    model_directory = os.path.join(
        config["checkpoint"],
    )
    tridet_model = TriDet.load_from_checkpoint(
        model_directory, map_location=config["device"]
    )
    tridet_model.model.eval()

    n_labelled_segments = 0
    n_visualisations = 0
    accuracies = []
    segment_lengths = []
    precision = []
    precision_start = []
    precision_end = []
    instructions = []

    pbar = tqdm.tqdm(total=config["n_labelled_segments"])
    while n_labelled_segments < config["n_labelled_segments"]:
        data = dataset.sample_segment()
        if data is None:
            continue
        obs, states, actions, rewards, directions, gt_cls = data

        obs_length = torch.tensor([obs.shape[0]]).long().to(config["device"])
        pred_segment, pred_label = tridet_model.segment_segment(
            [obs.to(config["device"]).unsqueeze(0).float(), obs_length, None, None]
        )

        start, end, valid = postprocess_predictions(pred_segment, obs_length)

        if not valid:
            continue

        # Add samples
        labelled_dataset["images"].append(
            blosc.pack_array(obs[start:end].cpu().numpy())
        )
        labelled_dataset["states"].append(blosc.pack_array(states[start:end]))
        labelled_dataset["actions"].append(actions[start : end - 1])
        labelled_dataset["rewards"].append(rewards[start : end - 1])
        labelled_dataset["directions"].append(directions[start : end - 1])
        labelled_dataset["instructions"].append(cl2inst[pred_label.item()])
        labelled_dataset["instruction_types"].append(cl2inst_type[pred_label.item()])

        # Logging info
        starts = [0]
        ends = [obs.shape[0]]
        prev_cl = gt_cls[0]
        for i, cl in enumerate(gt_cls[1:]):
            if cl != prev_cl:
                starts.append(i)
                ends.append(i)
                prev_cl = cl

        start_precision = start in starts
        end_precision = end in ends

        segment_lengths.append(end - start)
        precision += [start_precision, end_precision]
        precision_start.append(start_precision)
        precision_end.append(end_precision)
        instructions.append(cl2inst[pred_label.item()])

        # Visualize Segmentations
        if n_visualisations < config["n_visualisations"]:
            instruction = cl2inst[pred_label.item()]
            create_babyai_video(
                states[start:end], instruction, vis_directory, n_visualisations
            )
            n_visualisations += 1

        # Compute Accuracy
        accuracies += [gt == pred_label.item() for gt in gt_cls[start:end]]

        n_labelled_segments += 1
        pbar.update(1)

    # Log accuracy
    acc = np.array(accuracies).mean()

    wandb.log({"accuracy": acc})
    wandb.log({"Avg. Length": np.array(segment_lengths).mean()})
    wandb.log({"Precision": np.array(precision).mean()})
    wandb.log({"Precision Start": np.array(precision_start).mean()})
    wandb.log({"Precision End": np.array(precision_end).mean()})

    # Plot histogram of segment lengths
    segment_length_types = defaultdict(int)
    for segment_length in segment_lengths:
        segment_length_types[segment_length] += 1

    plt.bar(segment_length_types.keys(), segment_length_types.values())
    plt.xlabel("Segment Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"tridet_segment_length.png")
    plt.savefig(f"tridet_segment_length.pdf")

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
    plt.savefig(f"tridet_instruction_types.png")
    plt.savefig(f"tridet_instruction_types.pdf")

    # Save dataset
    filename = f"{config['labelled_dataset_file'].split('.')[0]}_tridet_{config['n_labelled_segments']}.pkl"
    tridet_dataset_file = os.path.join(config["dataset_directory"], filename)
    with open(tridet_dataset_file, "wb") as file:
        pickle.dump(labelled_dataset, file)


def postprocess_predictions(
    pred_segment: torch.Tensor, obs_length: int
) -> tuple[int, int, bool]:
    """
    Postprocesses the predicted segment by checking if it is valid and within the observed length.

    Args:
        pred_segment (torch.Tensor): The predicted segment as a tensor of shape (2,).
        obs_length (int): The length of the observed trajectory.

    Returns:
        tuple: A tuple containing the start and end indices of the segment, and a boolean indicating if the segment is valid.
    """
    start = int(pred_segment[0].item())
    end = int(pred_segment[1].item())

    if 0 <= start and end <= obs_length and start < end:
        return start, end, True

    return 0, 0, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    # Integrate CL arguments
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint

    # Set up logging
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"],
        dir="../results",
    )

    label_tridet(config)
