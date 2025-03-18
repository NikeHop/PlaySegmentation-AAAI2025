import argparse
import os

import lightning.pytorch as pl
import wandb
import yaml

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng

from play_segmentation.labelling.labelling_training.data import get_data
from play_segmentation.labelling.labelling_training.trainer import I3D


def train(config: dict) -> None:
    """
    Trains a model using the provided configuration.

    Args:
        config (dict): The configuration for training the model.

    Returns:
        None
    """
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
            model = I3D.load_from_checkpoint(
                os.path.join(model_directory, config["model"]["load"]["checkpoint"])
            )
        else:
            model = I3D(config["trainer"])

        # Get Data
        train_dataloader, test_dataloader = get_data(config["data"])

        # Create Trainer
        if config["training"]["distributed"]:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                max_epochs=config["training"]["epochs"],
                logger=logger,
                accelerator=config["training"]["accelerator"],
                devices=config["training"]["gpus"],
                strategy=config["training"]["strategy"],
                precision=config["training"]["precision"],
                accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
            )
        else:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                accelerator=config["training"]["accelerator"],
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
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    # Setup wandb project
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=["Pretraining"] + config["logging"]["tags"],
        name=f"Pretraining_{config['logging']['experiment_name']}",
        dir="../results",
    )

    wandb.config.update(config)

    train(config)
