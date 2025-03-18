""" Evaluate Imitation Learning Agent on BabyAI Environment. """

import argparse
import os

from collections import defaultdict

import gymnasium as gym
import wandb
import yaml

from torchvision.io import write_video

from play_segmentation.data.babyai.utils import CustomWrapper
from play_segmentation.policy.trainer import IL
from play_segmentation.utils.util import seeding


def evaluate_babyai(model: IL, config: dict) -> None:
    """
    Evaluate the performance of a BabyAI model.

    Args:
        model (object): The BabyAI model to evaluate.
        config (dict): Configuration parameters for the evaluation.

    Returns:
        None
    """

    if config["use_state"]:
        obs_type = "state"
    else:
        obs_type = "image"

    # Save Directory for Videos
    save_directory = config["save_directory"]
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Set seed
    seeding(config["seed"])

    # Create environment
    if config["env_name"] == "go_to":
        envs = [
            gym.make(
                f"BabyAI-GoToLocalS8N7-v0",
                highlight=False,
                max_steps=config["max_timesteps"],
            )
            for _ in range(config["num_envs"])
        ]
    else:
        raise NotImplementedError(f"This environment has not been implemented")

    envs = [CustomWrapper(env) for env in envs]

    # Prepare Model
    model.load_instruction_embeddings(config["embedding_config"])
    model.eval()

    # Prepare statistics
    total_rewards = defaultdict(list)
    completion_rate = defaultdict(list)
    trajectories = defaultdict(list)

    for k in range(config["n_iterations"]):
        # Reset the environments
        all_done = False
        n_timesteps = 0
        dones = []
        obss = []

        for env in envs:
            obs, _ = env.reset()
            obss.append(obs)

        while not all_done:
            new_obss = []
            actions = model.act(obss, obs_type)

            for i, (action, env) in enumerate(zip(actions, envs)):
                obs, reward, done, _, _ = env.step(action)
                dones.append(done)
                completion_rate[(k, i)].append(done)
                total_rewards[(k, i)].append(reward)
                new_obss.append(obs)
                trajectories[(k, i)].append(obs)

            # Check whether we are done
            all_done = all(dones)
            obss = new_obss
            n_timesteps += 1

            if n_timesteps > config["max_timesteps"]:
                all_done = True

        # Visualize the trajectories
        for i in range(min(10, config["num_envs"])):
            video = []
            for obs in trajectories[(0, i)]:
                img = obs["image"]
                video.append(img)
                instruction = obs["mission"]

            filename = f"{instruction}_{i}.mp4"

            visdir = config["save_directory"]
            if not os.path.exists(visdir):
                os.makedirs(visdir)
            filepath = os.path.join(visdir, filename)

            write_video(filepath, video, fps=2)

    rewards = []
    completions = []
    for j in range(config["n_iterations"]):
        for i in range(config["num_envs"]):
            rewards.append(max(total_rewards[(j, i)]))
            completions.append(any(completion_rate[(j, i)]))

    # Log statistics to WandDB
    wandb.log({f"evaluation/average_reward": sum(rewards) / len(rewards)})
    wandb.log({f"evaluation/success_rate": sum(completions) / len(completions)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/eval_babyai.yaml",
        help="Location of config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Location of checkpoint"
    )
    args = parser.parse_args()

    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    if args.checkpoint is not None:
        config["checkpoint"] = args.checkpoint

    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"],
        dir="../results",
    )

    evaluate_babyai(config)
