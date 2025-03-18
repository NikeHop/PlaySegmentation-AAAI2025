""" Project Utilities """

import random

import numpy as np
import torch


# Seed everything
def seeding(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Parameters:
        seed (int): The seed value to set.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
