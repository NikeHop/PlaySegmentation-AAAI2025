"""Training script for the different segmentation models"""

import argparse
import os

from typing import Callable

import lightning.pytorch as pl
import wandb
import yaml

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng

from play_segmentation.segmentation_models.unLoc.trainer import UnLoc
from play_segmentation.segmentation_models.unLoc.data import get_data_unloc
from play_segmentation.segmentation_models.tridet.data import get_data_tridet
from play_segmentation.segmentation_models.tridet.trainer import TriDet
from play_segmentation.segmentation_models.play_segment.data import (
    get_data_play_segmentation,
)
from play_segmentation.segmentation_models.play_segment.trainer import (
    PlaySegmentation,
)
from play_segmentation.utils.util import seeding


def get_baseline(
    config: dict,
) -> tuple[UnLoc | TriDet | PlaySegmentation, Callable]:
    """
    Get the baseline model and data loading function based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing the algorithm name.

    Returns:
        tuple: A tuple containing the baseline model class and the corresponding data loading function.

    Raises:
        NotImplementedError: If the specified algorithm is not implemented yet.
    """
    if config["algorithm"] == "unloc":
        baseline = UnLoc
        get_data = get_data_unloc
    elif config["algorithm"] == "tridet":
        baseline = TriDet
        get_data = get_data_tridet
    elif config["algorithm"] == "play_segmentation":
        baseline = PlaySegmentation
        get_data = get_data_play_segmentation
    else:
        raise NotImplementedError(
            f"This baseline ({config['algorithm']}) has not been implemented yet"
        )

    return baseline, get_data


def train(config: dict) -> None:
    """
    Train a segmentation model using the provided configuration.

    Args:
        config (dict): The configuration dictionary containing the training parameters.

    Returns:
        None
    """
    baseline, get_data = get_baseline(config)

    with isolate_rng():
        # Model Directory
        model_directory = os.path.join(
            config["logging"]["model_directory"], config["logging"]["experiment_name"]
        )
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Set up Logger
        logger = WandbLogger(
            project=config["logging"]["project"], save_dir=model_directory
        )

        # Create Model
        if config["trainer"]["load"]["load"]:
            model = baseline.load_from_checkpoint(
                os.path.join(model_directory, config["model"]["load"]["checkpoint"])
            )
        else:
            model = baseline(config["trainer"])

        # Get Data
        train_dataloader, test_dataloader = get_data(config["data"])

        # Create Trainer
        if config["training"]["distributed"]:
            if config["training"]["accelerator"] == "gpu":
                trainer = pl.Trainer(
                    default_root_dir=model_directory,
                    max_epochs=config["training"]["epochs"],
                    logger=logger,
                    accelerator=config["training"]["accelerator"],
                    devices=config["training"]["gpus"],
                    strategy=config["training"]["strategy"],
                    check_val_every_n_epoch=config["training"]["evaluation_frequency"],
                )
            else:
                trainer = pl.Trainer(
                    default_root_dir=model_directory,
                    max_epochs=config["training"]["epochs"],
                    logger=logger,
                    accelerator="cpu",
                )
        else:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                accelerator=config["training"]["accelerator"],
                devices=config["training"]["gpus"],
                max_epochs=config["training"]["epochs"],
                logger=logger,
                check_val_every_n_epoch=config["training"]["evaluation_frequency"],
            )

        # Training
        trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="path to experiment config file",
    )
    parser.add_argument(
        "--id3_checkpoint", default=None, type=str, help="ID3 checkpoint"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    # Integrate CLI arguments
    if args.id3_checkpoint:
        if config["algorithm"] == "tridet":
            config["trainer"]["model"]["preprocessor"][
                "checkpoint"
            ] = args.id3_checkpoint

        elif config["algorithm"] == "play_segmentation":
            config["trainer"]["model"]["i3d"]["checkpoint"] = args.id3_checkpoint

    # Setup wandb project
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=["Segmentation"] + config["logging"]["tags"],
        name=f"Segmentation_{config['logging']['experiment_name']}",
        dir="../results",
    )

    wandb.config.update(config)
    seeding(config["seed"])
    train(config)
