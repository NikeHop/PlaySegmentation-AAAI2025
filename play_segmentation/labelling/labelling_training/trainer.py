import lightning.pytorch as pl
import torch
import torch.nn as nn

from torch.optim import Adam

from play_segmentation.labelling.labelling_training.model import I3DModel


class I3D(pl.LightningModule):
    """
    I3D lightning trainer for video classification.

    Args:
        config (dict): Configuration parameters for the model.

    Attributes:
        model (I3DModel): The I3D model.
        lr (float): Learning rate for the optimizer.
        loss (nn.CrossEntropyLoss): Loss function for training.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.model = I3DModel(config["model"])
        self.lr = config["lr"]
        self.loss = nn.CrossEntropyLoss(reduce="mean")
        self.save_hyperparameters()

    def step(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and calculate the loss and accuracy.

        Args:
            batch (tuple): A tuple containing videos and labels.

        Returns:
            tuple: A tuple containing the loss and accuracy.
        """
        videos, _, _, labels = batch
        preds = self.model(videos)
        loss = self.loss(preds, labels)
        pred_labels = torch.argmax(preds, dim=-1)
        acc = (pred_labels == labels).float().mean()
        return loss, acc

    def training_step(self, batch: tuple, batch_id: int) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch: A batch of training data.
            batch_id: The index of the batch.

        Returns:
            float: The loss value.
        """
        loss, acc = self.step(batch)
        self.log("training/loss", loss)
        self.log("training/acc", acc)
        return loss

    def validation_step(self, batch: tuple, batch_id: int) -> None:
        """
        Perform a validation step.

        Args:
            batch: A batch of validation data.
            batch_id: The index of the batch.
        """
        self.step(batch)
        loss, acc = self.step(batch)
        self.log("validation/loss", loss)
        self.log("validation/acc", acc)

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Adam: The Adam optimizer.
        """
        return Adam(self.model.parameters(), lr=self.lr)

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encode the videos using the I3D model.

        Args:
            videos: The input videos.

        Returns:
            torch.Tensor: The encoded videos.
        """
        return self.model.encode(videos)

    def label(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Predict labels for the input videos.

        Args:
            videos: The input videos.

        Returns:
            torch.Tensor: The predicted labels.
        """
        preds = self.model(videos)
        return preds.argmax(dim=-1)
