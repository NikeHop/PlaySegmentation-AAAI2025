""" Training loop for MCIL policy """

import argparse
import os
import yaml

import lightning.pytorch as pl
import wandb

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng

from play_segmentation.policy.data import get_data
from play_segmentation.policy.trainer import MCIL
from play_segmentation.utils.util import seeding


def train_mcil(config: dict) -> None:
    """
    Trains the MCIL model using the provided configuration.

    Args:
        config (dict): The configuration for training the MCIL model.

    Returns:
        None
    """
    with isolate_rng():
        # Setup saving model directory
        model_directory = os.path.join(
            config["logging"]["model_directory"], config["logging"]["experiment_name"]
        )
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Set up Logger
        logger = WandbLogger(
            project=config["logging"]["project"], save_dir=model_directory
        )

        # Get Data
        train_dataloader, test_dataloader = get_data(config["data"])

        # Create Model
        if config["trainer"]["load"]["load"]:
            model = MCIL.load_from_checkpoint(
                os.path.join(
                    config["logging"]["model_directory"],
                    config["trainer"]["load"]["experiment_name"],
                    config["trainer"]["load"]["run_id"],
                    config["trainer"]["load"]["checkpoint"],
                )
            )
        else:
            model = MCIL(config["trainer"])

        # Create Trainer
        if config["training"]["distributed"]:
            if config["training"]["accelerator"] == "gpu":
                trainer = pl.Trainer(
                    default_root_dir=model_directory,
                    max_epochs=config["training"]["epochs"],
                    logger=logger,
                    accelerator="gpu",
                    devices=config["training"]["gpus"],
                    strategy=config["training"]["strategy"],
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
                max_epochs=config["training"]["epochs"],
                logger=logger,
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

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=int, default=None, help="Which gpu to run on")

    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    # Update config info
    if args.seed:
        config["seed"] = args.seed

    if args.device:
        config["training"]["gpus"] = [args.device]

    # Setup wandb project
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"],
        dir="../results",
    )

    wandb.config.update(config)
    seeding(config["seed"])
    train_mcil(config)
