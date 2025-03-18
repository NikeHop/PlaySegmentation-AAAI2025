""" Training loop for an imitation learning policy on BabyAI dataset """

import argparse
import os
import yaml

import lightning.pytorch as pl
import wandb

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng

from play_segmentation.policy.data import get_data
from play_segmentation.policy.trainer import IL
from play_segmentation.policy.eval import evaluate_babyai
from play_segmentation.utils.util import seeding


def train_iml(config: dict) -> None:
    """
    Train the imitation learning policy on BabyAI.

    Args:
        config (dict): Configuration parameters for training.

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
        train_dataloader, valid_dataloader = get_data(config["data"])

        # Create Model
        if config["trainer"]["load"]["load"]:
            model = IL.load_from_checkpoint(
                os.path.join(
                    config["logging"]["model_directory"],
                    config["trainer"]["load"]["experiment_name"],
                    config["trainer"]["load"]["run_id"],
                    config["trainer"]["load"]["checkpoint"],
                )
            )
        else:
            model = IL(config["trainer"])

        # Create Trainer
        if config["training"]["distributed"]:
            if config["training"]["accelerator"] == "gpu":
                trainer = pl.Trainer(
                    default_root_dir=model_directory,
                    max_steps=config["training"]["max_steps"],
                    logger=logger,
                    accelerator="gpu",
                    devices=config["training"]["gpus"],
                    strategy=config["training"]["strategy"],
                    val_check_interval=config["training"]["val_check_interval"],
                    check_val_every_n_epoch=None,
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
        trainer.fit(model, train_dataloader, valid_dataloader)

        # Perform evaluation
        save_directory = os.path.join(
            config["logging"]["model_directory"],
            config["logging"]["experiment_name"],
            wandb.run.id,
            "evaluation_videos",
        )
        config["evaluation"]["save_directory"] = save_directory
        evaluate_babyai(model, config["evaluation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="path to experiment config file",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed for reproducibility"
    )
    parser.add_argument("--device", type=int, default=None, help="device to train on")
    parser.add_argument(
        "--dataset-file", type=str, default=None, help="dataset to train on"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    # Overwrite config arguments
    if args.seed != None:
        config["seed"] = args.seed
    if args.device != None:
        config["training"]["gpus"] = [args.device]
    if args.dataset_file != None:
        config["data"]["dataset_file"] = args.dataset_file

    # Setup wandb project
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"],
        dir="../results",
    )

    wandb.config.update(config)

    # Setting all seeds
    seeding(config["seed"])

    train_iml(config)
