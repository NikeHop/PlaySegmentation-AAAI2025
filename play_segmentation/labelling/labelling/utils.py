""" Utilities for Labelled Segmentations. """

import os

from collections import defaultdict

import numpy as np
import torch
import tqdm

from torchvision.io import write_video
from PIL import Image, ImageDraw, ImageFont

from play_segmentation.data.babyai.utils import state2img


def get_frame(idx: int, episode_directory: str) -> dict:
    """
    Load and return a frame of the CALVIN environment.

    Parameters:
    - idx (int): The index of the frame to load.
    - episode_directory (str): The directory containing the episode files.

    Returns:
    - numpy.ndarray: The loaded frame.
    """

    idx = "0" * (7 - len(str(idx))) + str(idx)
    filename = os.path.join(episode_directory, f"episode_{idx}.npz")
    return np.load(filename)


def load_data(indeces: tuple[int, int], episode_directory: str) -> dict:
    """
    Load data from a given episode directory.

    Args:
        indeces (tuple): A tuple containing the start and end indices of the data to load.
        episode_directory (str): The directory path of the episode.

    Returns:
        dict: A dictionary containing the loaded data with the following keys:
            - "rgb_obs": A torch tensor of the "rgb_static" data, converted to float.
            - "gripper_obs": A torch tensor of the "rgb_gripper" data, converted to float.
            - "actions": A torch tensor of the "rel_actions" data, converted to float.
            - "clip_pretrained": The "clip_pretrained" data as a numpy array.
            - "clip_img_seq": A torch tensor of the "clip_pretrained" data, converted to float.
    """
    start, end = indeces
    data = defaultdict(list)
    data_keys = ["rgb_static", "clip_pretrained", "rel_actions", "rgb_gripper"]

    for i in range(start, end + 1):
        frame_data = get_frame(i, episode_directory)
        try:
            for key in data_keys:
                data[key].append(frame_data[key])
        except:
            print(f"Data in file {i} cannot be loaded")
            return None

    for key in data_keys:
        data[key] = np.stack(data[key], axis=0)

    return {
        "rgb_obs": torch.tensor(data["rgb_static"]).float(),
        "gripper_obs": torch.tensor(data["rgb_gripper"]).float(),
        "actions": torch.tensor(data["rel_actions"]).float(),
        "clip_pretrained": data["clip_pretrained"],
        "clip_img_seq": torch.from_numpy(data["clip_pretrained"]).float(),
    }


def preprocess_sample(sample: dict, config: dict) -> torch.Tensor:
    """
    Preprocesses a sample by converting the RGB observations to a video tensor.

    Args:
        sample (dict): The sample containing the RGB observations.
        config (dict): The configuration settings.

    Returns:
        torch.Tensor: The preprocessed video tensor.
    """
    video = sample["rgb_obs"].unsqueeze(0).to(config["device"])
    return video


def create_babyai_video(
    states: list, instruction: str, visualisation_directory: str, idx: int
) -> None:
    """
    Create a video from a sequence of states in the BabyAI environment.

    Args:
        states (list): A list of states.
        instruction (str): The instruction for the video.
        visualisation_directory (str): The directory to save the video.
        idx (int): The index of the video.

    Returns:
        None
    """
    video = []
    for state in states:
        img = state2img(state)
        video.append(img)

    video = np.stack(video, axis=0)
    torch.from_numpy(video)

    videoname = f"{instruction}_{idx}.mp4"
    videofile = os.path.join(visualisation_directory, videoname)
    write_video(videofile, video, fps=10)


def create_video_with_text(
    video_array: np.ndarray,
    descriptions: dict,
    output_path: str,
    fps: int = 5,
    added_height: int = 50,
) -> None:
    """
    Create an .mp4 video from a NumPy array while displaying text below the image.

    Parameters:
    - video_array: NumPy array of shape (T, H, W, 3) where T is time, HxW is resolution, and 3 is color channels.
    - descriptions: Dictionary {timestep: text} mapping specific frames to text descriptions.
    - output_path: Path to save the output video.
    - fps: Frames per second.
    - text_height: Height of the whitespace area for text.
    """
    T, H, W, _ = video_array.shape
    new_H = H + added_height  # Increased height for the text area
    frames = []

    for t in range(T):
        frame = video_array[t].astype(np.uint8)

        # Create a blank white bar for text
        text_bar = (
            np.ones((added_height, W, 3), dtype=np.uint8) * 255
        )  # White background

        # Convert to PIL Image for text overlay
        image = Image.fromarray(text_bar)
        draw = ImageDraw.Draw(image)

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        # Get text for the current timestep
        text = get_text(descriptions, t)
        if text:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (W - text_width) // 2  # Center horizontally
            text_y = (text_bar.shape[0] - text_height) // 2  # Center vertically
            draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))  # Black text

        # Convert back to NumPy
        text_bar = np.array(image)

        # Stack original frame with text bar
        frame_with_text = np.vstack((frame, text_bar))
        frames.append(frame_with_text)

    video = np.stack(frames, axis=0)
    video = torch.from_numpy(video)
    write_video(output_path, video, fps=fps)


def get_text(descriptions: dict, t: int) -> str:
    """
    Retrieve the text description for a given time point.

    Args:
        descriptions (dict): A dictionary containing time points as keys and corresponding descriptions as values.
        t (int): The time point for which to retrieve the description.

    Returns:
        str: The text description for the given time point, or "No Instruction Found" if no description is found.
    """
    for key, value in sorted(descriptions.items(), key=lambda x: x[0]):
        if t < key:
            return value
    else:
        return "No Instruction Found"
