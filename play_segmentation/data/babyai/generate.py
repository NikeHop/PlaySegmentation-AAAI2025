"""Functionalities for generating babyai dataset"""

import argparse
import os
import pickle
import random
import multiprocessing

from collections import defaultdict
from multiprocessing import cpu_count

import blosc
import numpy as np
import torch
import tqdm
import warnings
import yaml

from minigrid.utils.baby_ai_bot import BabyAIBot

from play_segmentation.data.babyai.env import GoToLocalExtended
from play_segmentation.data.babyai.utils import (
    instrs2action,
    InstructionType,
    CustomWrapper,
)
from play_segmentation.utils.util import seeding


def generate_consecutive_instruction_episodes(data: tuple) -> tuple:
    """
    Generate consecutive instruction episodes for a given environment.

    Args:
        data (tuple): A tuple containing the environment, seed, and configuration.

    Returns:
        tuple: A tuple containing the complete images, instructions, instruction types,
        complete actions, complete directions, and complete rewards.
    """
    env, seed, config = data
    generated_trajectory = False
    n_resamples = 0
    while not generated_trajectory:
        # Data for a single unsegmented trajectory
        instructions = []
        instruction_types = []
        complete_actions = []
        complete_images = []
        complete_states = []
        complete_directions = []
        complete_rewards = []

        n_steps = 0
        curr_seed = seed
        obs = env.reset(seed=curr_seed)[0]

        old_goal_object = None
        old_goal_colour = None

        for _ in range(config["max_n_instructions"]):
            no_instruction = True
            for _ in range(10):
                try:
                    env.env.sample_new_instruction()
                    no_instruction = False
                    break
                except Exception as e:
                    print("Failed to sample new instruction")
                    continue

            # Check whether the goal object is not the same as before
            new_goal_object = env.unwrapped.instrs.desc.type
            new_goal_colour = env.unwrapped.instrs.desc.color
            while (
                old_goal_object == new_goal_object
                and old_goal_colour == new_goal_colour
            ):
                for _ in range(10):
                    try:
                        env.env.sample_new_instruction()
                        no_instruction = False
                        break
                    except Exception:
                        print("Failed to sample new instruction")
                        continue
                new_goal_object = env.unwrapped.instrs.desc.type
                new_goal_colour = env.unwrapped.instrs.desc.color

            if no_instruction:
                break

            old_goal_colour = new_goal_colour
            old_goal_object = new_goal_object
            mission = env.unwrapped.mission
            action = instrs2action(env.unwrapped.instrs)
            obj = env.unwrapped.instrs.desc.type
            color = env.unwrapped.instrs.desc.color
            location = env.unwrapped.instrs.desc.loc
            instruction_type = InstructionType(action, obj, color, location)
            actions = []
            images = []
            states = []
            directions = []
            rewards = []
            done = False
            mission_success = False
            agent = BabyAIBot(env)
            n_steps = 0

            # Complete current task
            while not done and n_steps < config["max_steps"]:
                n_steps += 1
                try:
                    action = agent.replan()
                except Exception as e:
                    print("Replanning failed")
                    break
                if isinstance(action, torch.Tensor):
                    action = action.item()

                new_obs, reward, done, _, _ = env.step(action)

                if done and reward > 0:
                    mission_success = True

                actions.append(action)
                images.append(obs["image"])
                states.append(obs["state"])
                directions.append(obs["direction"])
                rewards.append(reward)
                n_steps += 1
                obs = new_obs

            # If our demos was successful, save it
            if mission_success:
                images.append(obs["image"])
                states.append(obs["state"])
                instructions.append(mission)
                instruction_types.append(instruction_type.to_dict())
                complete_actions.append(actions)
                complete_images.append(blosc.pack_array(np.array(images)))
                complete_states.append(blosc.pack_array(np.array(states)))
                complete_directions.append(directions)
                complete_rewards.append(rewards)

            else:
                print("Mission failed")
                break

        if len(instructions) >= config["min_n_instructions"]:
            generated_trajectory = True
        else:
            seed = random.randint(1000000, 2000000)
            n_resamples += 1
            if n_resamples > 10:
                warnings.warn("Failed to generate a trajectory")
                return None

    return (
        complete_images,
        complete_states,
        instructions,
        instruction_types,
        complete_actions,
        complete_directions,
        complete_rewards,
    )


def generate_trajectory_datasets(config: dict) -> None:
    """
    Generate trajectory datasets based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for generating the datasets.

    Returns:
        None
    """
    # Seed everything
    seeding(config["seed"])

    # Check whether save directory exists
    if not os.path.exists(config["save_directory"]):
        os.makedirs(config["save_directory"])

    # Create environment
    env = GoToLocalExtended(
        highlight=False,
        agent_view_size=config["room_size"] + 1,
        room_size=config["room_size"],
        num_dists=config["num_dists"],
    )
    env = CustomWrapper(env)

    # Generate trajectories via multiprocessing
    dataset = defaultdict(list)
    n_trajectories = 0
    pbar = tqdm.tqdm(total=config["n_trajectories"])
    timeout = 10
    with multiprocessing.Pool(processes=cpu_count()) as pool:
        # Submit all tasks asynchronously
        results = [
            pool.apply_async(
                generate_consecutive_instruction_episodes, ((env, seed, config),)
            )
            for seed in range(config["n_trajectories"])
        ]

        # Collect results with timeout
        for i, result in enumerate(results):
            try:
                trajectory = result.get(
                    timeout=timeout
                )  # Wait for the result with a timeout
                if trajectory is None:
                    continue

                (
                    traj_images,
                    traj_states,
                    traj_instructions,
                    traj_instruction_types,
                    traj_actions,
                    traj_directions,
                    traj_rewards,
                ) = trajectory

                dataset["images"].append(traj_images)
                dataset["states"].append(traj_states)
                dataset["instructions"].append(traj_instructions)
                dataset["instruction_types"].append(traj_instruction_types)
                dataset["actions"].append(traj_actions)
                dataset["directions"].append(traj_directions)
                dataset["rewards"].append(traj_rewards)

                pbar.update()

            except multiprocessing.TimeoutError:
                print(f"Task timed out")

    new_dataset = defaultdict(list)
    n_trajectories = 0
    for (
        imgs,
        states,
        instructions,
        instruction_types,
        actions,
        directions,
        rewards,
    ) in zip(
        dataset["images"],
        dataset["states"],
        dataset["instructions"],
        dataset["instruction_types"],
        dataset["actions"],
        dataset["directions"],
        dataset["rewards"],
    ):
        new_dataset["images"].append(imgs)
        new_dataset["states"].append(states)
        new_dataset["instructions"].append(instructions)
        new_dataset["instruction_types"].append(instruction_types)
        new_dataset["actions"].append(actions)
        new_dataset["directions"].append(directions)
        new_dataset["rewards"].append(rewards)
        n_trajectories += len(imgs)
        if n_trajectories >= config["n_trajectories"]:
            break

    dataset = new_dataset

    # Save unsegmented trajectory dataset
    n = len(dataset["images"])
    validation_indices = set(
        random.sample(list(range(n)), k=int(config["validation_ratio"] * n))
    )

    validation_dataset = {
        "images": [],
        "states": [],
        "instructions": [],
        "instruction_types": [],
        "actions": [],
        "directions": [],
        "rewards": [],
    }
    training_dataset = {
        "images": [],
        "states": [],
        "instructions": [],
        "instruction_types": [],
        "actions": [],
        "directions": [],
        "rewards": [],
    }

    for i in range(n):
        if i in validation_indices:
            validation_dataset["images"].append(dataset["images"][i])
            validation_dataset["states"].append(dataset["states"][i])
            validation_dataset["instructions"].append(dataset["instructions"][i])
            validation_dataset["instruction_types"].append(
                dataset["instruction_types"][i]
            )
            validation_dataset["actions"].append(dataset["actions"][i])
            validation_dataset["directions"].append(dataset["directions"][i])
            validation_dataset["rewards"].append(dataset["rewards"][i])
        else:
            training_dataset["images"].append(dataset["images"][i])
            training_dataset["states"].append(dataset["states"][i])
            training_dataset["instructions"].append(dataset["instructions"][i])
            training_dataset["instruction_types"].append(
                dataset["instruction_types"][i]
            )
            training_dataset["actions"].append(dataset["actions"][i])
            training_dataset["directions"].append(dataset["directions"][i])
            training_dataset["rewards"].append(dataset["rewards"][i])

    with open(
        os.path.join(
            config["save_directory"],
            f"babyai_{config['env_name']}_{config['n_trajectories']}_unsegmented_training.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(training_dataset, file)

    with open(
        os.path.join(
            config["save_directory"],
            f"babyai_{config['env_name']}_{config['n_trajectories']}_unsegmented_validation.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(validation_dataset, file)

    # Convert unsegmented to segmented dataset
    training_segmented_dataset = {}

    for key, value in dataset.items():
        flatten_value = []
        for val in value:
            flatten_value.extend(val)
        training_segmented_dataset[key] = flatten_value

    validation_segmented_dataset = {}

    for key, value in dataset.items():
        flatten_value = []
        for val in value:
            flatten_value.extend(val)
        validation_segmented_dataset[key] = flatten_value

    # Save dataset of individual segments
    training_segmented_filename = (
        f"babyai_{config['env_name']}_{config['n_trajectories']}_single_training.pkl"
    )
    with open(
        os.path.join(config["save_directory"], training_segmented_filename), "wb"
    ) as file:
        pickle.dump(training_segmented_dataset, file)

    validation_segmented_filename = (
        f"babyai_{config['env_name']}_{config['n_trajectories']}_single_validation.pkl"
    )
    with open(
        os.path.join(config["save_directory"], validation_segmented_filename), "wb"
    ) as file:
        pickle.dump(validation_segmented_dataset, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Location of the config file")
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=None,
        help="Number of trajectories to generate",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update config with CL arguments
    if args.n_trajectories != None:
        config["n_trajectories"] = args.n_trajectories

    generate_trajectory_datasets(config)
