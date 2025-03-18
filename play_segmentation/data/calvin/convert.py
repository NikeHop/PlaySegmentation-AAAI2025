import argparse
import glob
import os
import random

from collections import defaultdict
from PIL import Image

import clip
import numpy as np
import torch
import yaml

from tqdm import tqdm


def preprocess(config: dict) -> None:
    """
    Preprocesses the data for trajectory segmentation.

    Args:
        config (dict): Configuration parameters for the preprocessing.

    Returns:
        None
    """
    # Preprocess the annotations
    preprocess_annotations("calvin_debug_dataset", config)
    preprocess_annotations("task_D_D", config)

    # Embed the frames using CLIP for UnLOC
    preprocess_frames(config)


def preprocess_frames(config: dict) -> None:
    """
    Preprocesses frames using the CLIP model.

    Args:
        config (dict): Configuration parameters for preprocessing.

    Returns:
        None
    """
    # Load CLIP
    model, transform = clip.load(
        name=config["vlm"],
        device=config["device"],
        download_root=config["model_store"],
    )

    # Debug - Training
    training_directory = os.path.join(
        config["dataset_directory"], "calvin_debug_dataset", "training"
    )
    files = glob.glob(training_directory + "/*.npz")
    process_files(files, model, transform, config)

    # Debug - Validation
    validation_directory = os.path.join(
        config["dataset_directory"], "calvin_debug_dataset", "validation"
    )
    files = glob.glob(validation_directory + "/*.npz")
    process_files(files, model, transform, config)

    # Task D_D - Training
    training_directory = os.path.join(
        config["dataset_directory"], "task_D_D", "training"
    )
    files = glob.glob(training_directory + "/*.npz")
    process_files(files, model, transform, config)

    # Task D_D - Validation
    validation_directory = os.path.join(
        config["dataset_directory"], "task_D_D", "validation"
    )
    files = glob.glob(validation_directory + "/*.npz")
    process_files(files, model, transform, config)


@torch.no_grad()
def process_files(files: list[str], model, transform, config: dict) -> None:
    """
    Process a list of files by converting RGB images to CLIP embeddings and saving the results.

    Args:
        files (list[str]): List of file paths to process.
        model: The CLIP model used to encode the images.
        transform: The transformation applied to the RGB images.
        config (dict): Configuration parameters for the processing.

    Returns:
        None
    """
    count = 0
    pbar = tqdm(total=len(files))
    while count < len(files):
        files_to_process = files[count : count + config["batch_size"]]

        files_data = []
        for file in files_to_process:
            data = np.load(file, allow_pickle=True).items()

            files_data.append({key: value for key, value in data})

        # Process the batch of RGB-images to CLIP embeddings
        rgb_images = [
            transform(Image.fromarray(data["rgb_static"])) for data in files_data
        ]
        rgb_images = torch.stack(rgb_images, dim=0).to(config["device"])
        encoded_images = model.encode_image(rgb_images)

        # Add the clip embeddings to the data files
        for file, file_data, image in zip(files_to_process, files_data, encoded_images):
            file_data["clip_pretrained"] = image.cpu().numpy()
            with open(file, "wb") as file:
                np.savez(file, **file_data)

        count += config["batch_size"]
        pbar.update(config["batch_size"])


@torch.no_grad()
def preprocess_annotations(dataset_name: str, config: dict):
    """
    Preprocesses the annotations for a given dataset.

    Args:
        dataset_name (str): The name of the dataset.
        config (dict): The configuration settings.

    Returns:
        None
    """
    # Load CLIP model
    model, transform = clip.load(
        name=config["vlm"],
        device=config["device"],
        download_root=config["model_store"],
    )

    # Load data
    training_annotation_file = os.path.join(
        config["dataset_directory"],
        dataset_name,
        "training",
        f"lang_annotations/auto_lang_ann.npy",
    )
    training_annotations = np.load(training_annotation_file, allow_pickle=True).item()
    validation_annotation_file = os.path.join(
        config["dataset_directory"],
        dataset_name,
        "validation",
        f"lang_annotations/auto_lang_ann.npy",
    )
    validation_annotations = np.load(
        validation_annotation_file, allow_pickle=True
    ).item()

    for key in training_annotations:
        print(key)

    for key in training_annotations["language"]:
        print(key)

    for key in validation_annotations["info"]:
        print(key)

    """
    # Step 1
    Create a dictionary that maps the frames to the class labels.
    """
    all_tasks = (
        training_annotations["language"]["task"]
        + validation_annotations["language"]["task"]
    )
    all_tasks = list(sorted(set(all_tasks)))
    task2cl = {task: idd for idd, task in enumerate(all_tasks)}

    task_frames2row = {}
    for task, frame in zip(
        training_annotations["language"]["task"], training_annotations["info"]["indx"]
    ):
        task_frames2row[frame] = task2cl[task]
    training_annotations["language"]["task_frames2row"] = task_frames2row

    task_frames2row = {}
    for task, frame in zip(
        validation_annotations["language"]["task"],
        validation_annotations["info"]["indx"],
    ):
        task_frames2row[frame] = task2cl[task]
    validation_annotations["language"]["task_frames2row"] = task_frames2row

    """
    Step 2
    Map each task to a representative instruction
    """
    task2instructions = defaultdict(list)
    for task, instruction in zip(
        training_annotations["language"]["task"],
        training_annotations["language"]["ann"],
    ):
        task2instructions[task].append(instruction)

    for task, instruction in zip(
        validation_annotations["language"]["task"],
        validation_annotations["language"]["ann"],
    ):
        task2instructions[task].append(instruction)

    task2instruction = {
        task: list(sorted(instructions))[0]
        for task, instructions in task2instructions.items()
    }

    """
    # Step 3
    Embed the instructions via CLIP
    """

    instructions = [task2instruction[task] for task in set(all_tasks)]
    instruction2row = {instruction: row for row, instruction in enumerate(instructions)}
    instruction_tokens = clip.tokenize(instructions).to(config["device"])
    instruction_embeddings = model.encode_text(instruction_tokens).to("cpu").numpy()

    training_clip_frame2row = {}
    for frame, task in zip(
        training_annotations["info"]["indx"], training_annotations["language"]["task"]
    ):
        inst = task2instruction[task]
        row = instruction2row[inst]
        training_clip_frame2row[frame] = row

    validation_clip_frame2row = {}
    for frame, task in zip(
        validation_annotations["info"]["indx"],
        validation_annotations["language"]["task"],
    ):
        inst = task2instruction[task]
        row = instruction2row[inst]
        validation_clip_frame2row[frame] = row

    training_annotations["language"]["clip"] = instruction_embeddings
    training_annotations["language"]["clip_frames2row"] = training_clip_frame2row
    validation_annotations["language"]["clip"] = instruction_embeddings
    validation_annotations["language"]["clip_frames2row"] = validation_clip_frame2row

    """
    # Step 3
    Save the new annotations
    """
    with open(training_annotation_file, "wb") as file:
        np.save(file, training_annotations)

    with open(validation_annotation_file, "wb") as file:
        np.save(file, validation_annotations)

    if dataset_name == "task_D_D":
        """
        # Step 4
        Split the dataset into different percentages
        """
        N = len(training_annotations["language"]["task"])

        indeces_50 = random.sample(range(N), N // 2)
        indeces_25 = random.sample(indeces_50, N // 4)
        indeces_10 = random.sample(indeces_25, N // 10)

        indeces_50 = set(indeces_50)
        indeces_25 = set(indeces_25)
        indeces_10 = set(indeces_10)

        annotations_10 = {
            "language": {
                "task": [],
                "ann": [],
                "emb": [],
                "task_frames2row": [],
                "clip_frames2row": [],
                "clip": [],
            },
            "info": {"indx": []},
        }

        annotations_25 = {
            "language": {
                "task": [],
                "ann": [],
                "emb": [],
                "task_frames2row": [],
                "clip_frames2row": [],
                "clip": [],
            },
            "info": {"indx": []},
        }

        annotations_50 = {
            "language": {
                "task": [],
                "ann": [],
                "emb": [],
                "task_frames2row": [],
                "clip_frames2row": [],
                "clip": [],
            },
            "info": {"indx": []},
        }

        for i in range(N):
            if i in indeces_10:
                annotations_10["language"]["task"].append(
                    training_annotations["language"]["task"][i]
                )
                annotations_10["language"]["ann"].append(
                    training_annotations["language"]["ann"][i]
                )
                annotations_10["language"]["emb"].append(
                    training_annotations["language"]["emb"][i]
                )
                annotations_10["info"]["indx"].append(
                    training_annotations["info"]["indx"][i]
                )

            if i in indeces_25:
                annotations_25["language"]["task"].append(
                    training_annotations["language"]["task"][i]
                )
                annotations_25["language"]["ann"].append(
                    training_annotations["language"]["ann"][i]
                )
                annotations_25["language"]["emb"].append(
                    training_annotations["language"]["emb"][i]
                )
                annotations_25["info"]["indx"].append(
                    training_annotations["info"]["indx"][i]
                )

            if i in indeces_50:
                annotations_50["language"]["task"].append(
                    training_annotations["language"]["task"][i]
                )
                annotations_50["language"]["ann"].append(
                    training_annotations["language"]["ann"][i]
                )
                annotations_50["language"]["emb"].append(
                    training_annotations["language"]["emb"][i]
                )
                annotations_50["info"]["indx"].append(
                    training_annotations["info"]["indx"][i]
                )
            else:
                continue

        # Downsample frames2row
        annotations_10["language"]["task_frames2row"] = {
            frame: row
            for frame, row in training_annotations["language"][
                "task_frames2row"
            ].items()
            if frame in annotations_10["info"]["indx"]
        }
        annotations_10["language"]["clip_frames2row"] = {
            frame: row
            for frame, row in training_annotations["language"][
                "clip_frames2row"
            ].items()
            if frame in annotations_10["info"]["indx"]
        }
        annotations_10["language"]["clip"] = training_annotations["language"]["clip"]

        annotations_25["language"]["task_frames2row"] = {
            frame: row
            for frame, row in training_annotations["language"][
                "task_frames2row"
            ].items()
            if frame in annotations_25["info"]["indx"]
        }
        annotations_25["language"]["clip_frames2row"] = {
            frame: row
            for frame, row in training_annotations["language"][
                "clip_frames2row"
            ].items()
            if frame in annotations_25["info"]["indx"]
        }
        annotations_25["language"]["clip"] = training_annotations["language"]["clip"]

        annotations_50["language"]["task_frames2row"] = {
            frame: row
            for frame, row in training_annotations["language"][
                "task_frames2row"
            ].items()
            if frame in annotations_50["info"]["indx"]
        }
        annotations_50["language"]["clip_frames2row"] = {
            frame: row
            for frame, row in training_annotations["language"][
                "clip_frames2row"
            ].items()
            if frame in annotations_50["info"]["indx"]
        }
        annotations_50["language"]["clip"] = training_annotations["language"]["clip"]

        # Save the new annotations
        print(
            f"Saving 10% Dataset with {len(annotations_10['language']['task_frames2row'])} samples"
        )
        annotations_10_file = os.path.join(
            config["dataset_directory"],
            "task_D_D",
            "training",
            f"lang_annotations/auto_lang_ann_10.npy",
        )
        with open(annotations_10_file, "wb") as file:
            np.save(file, annotations_10)

        print(
            f"Saving 25% Dataset with {len(annotations_25['language']['task_frames2row'])} samples"
        )
        annotations_25_file = os.path.join(
            config["dataset_directory"],
            "task_D_D",
            "training",
            f"lang_annotations/auto_lang_ann_25.npy",
        )
        with open(annotations_25_file, "wb") as file:
            np.save(file, annotations_25)

        print(
            f"Saving 50% Dataset with {len(annotations_50['language']['task_frames2row'])} samples"
        )
        annotations_50_file = os.path.join(
            config["dataset_directory"],
            "task_D_D",
            "training",
            f"lang_annotations/auto_lang_ann_50.npy",
        )
        with open(annotations_50_file, "wb") as file:
            np.save(file, annotations_50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Location of the config file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    preprocess(config)
