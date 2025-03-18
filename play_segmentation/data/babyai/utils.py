"""Utilities for the BabyAI environment"""

from typing import Tuple

import numpy as np

from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from minigrid.envs.babyai.core.verifier import (
    PickupInstr,
    OpenInstr,
    GoToInstr,
    PutNextInstr,
)
from minigrid.core.grid import Grid
from minigrid.core.constants import COLORS, IDX_TO_COLOR

LOC_NAMES = ["left to", "right to", "in front of", "behind", None]
OBJ_TYPES = ["box", "ball", "key", "door"]
ACTIONS = ["Go to", "Pick up", "Open"]
COLOUR_NAMES = ["red", "green", "blue", "yellow", "purple", "grey"]


# Structured representation of possible instructions
class InstructionType:
    def __init__(
        self, action=None, obj=None, color=None, location=None, multiple=False
    ):
        self.action = action
        self.obj = obj
        self.color = color
        self.location = location
        self.multiple = multiple

    def to_dict(self):
        return {
            "action": self.action,
            "obj": self.obj,
            "color": self.color,
            "location": self.location,
            "multiple": self.multiple,
        }

    def from_dict(self, info):
        self.action = info["action"]
        self.obj = info["obj"]
        self.color = info["color"]
        self.location = info["location"]
        self.multiple = info["multiple"]

    def __str__(self):
        if self.multiple:
            article = "a"
        else:
            article = "the"

        instruction = self.action
        if self.color is not None:
            instruction += f" {article} {self.color}"
        else:
            instruction += f" {article}"

        instruction += f" {self.obj}"

        if self.location is not None and self.location != "":
            instruction += f" {self.location} you"

        return instruction.lower()

    def __gt__(self, inst_type2):
        return str(self) > str(inst_type2)


def generate_all_instructions():
    instructions = []
    for action in ACTIONS:
        if action == "Put":
            continue
        for loc in LOC_NAMES:
            for obj in OBJ_TYPES:
                for colour in COLOUR_NAMES:
                    if loc != "":
                        instructions.append(f"{action} the {colour} {obj} {loc} you")
                    else:
                        instructions.append(f"{action} the {colour} {obj}")

    return instructions


def instrs2action(instrs):
    if isinstance(instrs, GoToInstr):
        return "Go to"
    elif isinstance(instrs, PickupInstr):
        return "Pick up"
    elif isinstance(instrs, OpenInstr):
        return "Open"
    elif isinstance(instrs, PutNextInstr):
        return "Put"
    else:
        raise NotImplementedError("Instruction not recognized")


# Generate all possible instructions in the "Go-To"-environment
ALL_INSTRUCTIONS_GOTOLOCAL = []
ALL_INSTRUCTION_TYPES_GOTOLOCAL = []

for obj in OBJ_TYPES:
    if obj == "door":
        continue
    for colour in COLORS:
        for multiple in [True, False]:
            inst_type = InstructionType("Go to", obj, colour, "", multiple)
            ALL_INSTRUCTION_TYPES_GOTOLOCAL.append(inst_type)
            ALL_INSTRUCTIONS_GOTOLOCAL.append(str(inst_type))

CL2INSTRUCTION_TYPE_GOTOLOCAL = {
    i: inst_type for i, inst_type in enumerate(sorted(ALL_INSTRUCTION_TYPES_GOTOLOCAL))
}
CL2INSTRUCTION_GOTOLOCAL = {
    i: str(inst_type) for i, inst_type in CL2INSTRUCTION_TYPE_GOTOLOCAL.items()
}


class CustomWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env = RGBImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.unwrapped.height * tile_size,
                self.unwrapped.width * tile_size,
                3,
            ),
            dtype="uint8",
        )

        old_img_space = self.observation_space.spaces["image"]

        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "state": old_img_space,
                "image": new_image_space,
            }
        )

    def observation(self, obs):
        rgb_img = self.unwrapped.get_frame(
            highlight=self.unwrapped.highlight, tile_size=self.tile_size
        )
        state = self.unwrapped.grid.encode()
        i, j = self.unwrapped.agent_pos
        state[i, j, 0] = 10
        state[i, j, 2] = self.unwrapped.agent_dir

        return {**obs, "state": state, "image": rgb_img}


def extract_agent_pos_and_direction(obs: np.array) -> Tuple[int, int]:
    """
    Extracts the agent's position and direction from the observation

    Args:
        obs (np.array): environment observation
    """
    agent_pos = None
    agent_dir = None
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            if obs[i][j][0] == 10:
                agent_pos = (i, j)
                agent_dir = obs[i][j][2]
                break
    return agent_pos, agent_dir


def state2img(state: np.array) -> np.array:
    """
    Take in an environment observation and convert it to an RGB image

    Args:
        state (np.array): environment observation

    Returns:
        img (np.array): RGB image of the environment observation
    """
    agent_pos, agent_dir = extract_agent_pos_and_direction(state)
    grid, _ = Grid.decode(state)

    # States generated by the diffusion model do not necessarily have an agent
    if agent_pos is not None:
        agent_type = state[agent_pos[0]][agent_pos[1]][1]
    else:
        agent_type = 0

    agent_color = COLORS[IDX_TO_COLOR[agent_type]]
    img = grid.render(32, agent_pos, agent_dir)
    return img
