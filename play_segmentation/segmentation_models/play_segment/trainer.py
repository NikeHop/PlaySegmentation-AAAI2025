import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import Adam

from play_segmentation.segmentation_models.play_segment.model import (
    PlaySegmentationModel,
)


class PlaySegmentation(pl.LightningModule):
    """
    LightningModule for training and evaluating the PlaySegmentation model.

    Args:
        config (dict): Configuration parameters for the model.

    Attributes:
        lr (float): Learning rate for the optimizer.
        model (PlaySegmentationModel): The PlaySegmentation model.
        binary_loss (nn.BCELoss): Binary cross-entropy loss function.
        cls_loss (nn.CrossEntropyLoss): Cross-entropy loss function for classification.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["lr"]
        self.model = PlaySegmentationModel(config["model"])
        self.binary_loss = nn.BCELoss(reduce="mean")
        self.cls_loss = nn.CrossEntropyLoss(reduce="mean")

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Adam: The Adam optimizer.
        """
        return Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, data: dict) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            data (Any): The input data for the training step.

        Returns:
            torch.Tensor: The loss value for the training step.
        """
        seg_probs, cls_probs = self.model(data)  # BxT
        seg_loss, cls_loss, metrics = self._get_loss(seg_probs, cls_probs, data)
        loss = cls_loss + seg_loss

        self.log("training/loss", loss)
        self.log("training/cls_loss", cls_loss)
        self.log("training/seg_loss", seg_loss)

        for key, value in metrics.items():
            self.log(f"training/{key}", value)

        return loss

    def validation_step(self, data: dict) -> None:
        """
        Perform a validation step on the given data.

        Args:
            data: The input data for the validation step.

        Returns:
            None
        """
        seg_probs, cls_probs = self.model(data)
        seg_loss, cls_loss, metrics = self._get_loss(seg_probs, cls_probs, data)
        loss = seg_loss + cls_loss

        self.log("validation/seg_loss", seg_loss, sync_dist=True)
        self.log("validation/cls_loss", cls_loss, sync_dist=True)
        self.log("validation/loss", loss, sync_dist=True)
        for key, value in metrics.items():
            self.log(f"validation/{key}", value, sync_dist=True)

    def _get_loss(
        self, seg_probs: torch.Tensor, cls_probs: torch.Tensor, data: dict
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the segmentation and classification loss.

        Args:
            seg_probs (torch.Tensor): Segmentation probabilities.
            cls_probs (torch.Tensor): Classification probabilities.
            data (Any): The input data.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict]: A tuple containing the segmentation loss, classification loss, and metrics.
        """

        cls_mask = data["class_labels"] > -1
        cls_labels = data["class_labels"][cls_mask]
        cls_loss = self.cls_loss(cls_probs, cls_labels)
        seg_loss = self.binary_loss(seg_probs, data["labels"])
        metrics = self._get_metrics(seg_probs, cls_probs, data["labels"], cls_labels)

        return seg_loss, cls_loss, metrics

    def _get_metrics(
        self,
        seg_probs: torch.Tensor,
        cls_probs: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> dict:
        """
        Calculate various metrics for evaluating the segmentation and classification results.

        Args:
            seg_probs (torch.Tensor): Segmentation probabilities.
            cls_probs (torch.Tensor): Classification probabilities.
            gt_labels (torch.Tensor): Ground truth segmentation labels.
            gt_classes (torch.Tensor): Ground truth classification labels.

        Returns:
            dict: A dictionary containing the following metrics:
                - "accuracy": Segmentation accuracy.
                - "abs_diff": Absolute difference between segmentation probabilities and ground truth labels.
                - "cls_accuracy": Classification accuracy.
                - "termination_acc": Accuracy for 1s in the segmentation labels.
                - "none_termination_acc": Accuracy for 0s in the segmentation labels.
        """
        # Segmentation Accuracy
        predicted_labels = torch.bernoulli(seg_probs)
        acc = (gt_labels == predicted_labels).float().mean()
        abs_diff = torch.abs(seg_probs - gt_labels).mean()

        # Accuracy for 1s
        one_mask = gt_labels == 1
        termination_acc = (
            (gt_labels[one_mask] == predicted_labels[one_mask]).float().mean()
        )

        # Accuracy for labels
        zero_mask = gt_labels == 0
        no_termination_acc = (
            (gt_labels[zero_mask] == predicted_labels[zero_mask]).float().mean()
        )

        # Class Accuracy
        predicted_cls = cls_probs.argmax(dim=-1)
        cls_acc = (predicted_cls == gt_classes).float().mean()

        return {
            "accuracy": acc,
            "abs_diff": abs_diff,
            "cls_accuracy": cls_acc,
            "termination_acc": termination_acc,
            "none_termination_acc": no_termination_acc,
        }

    def get_log_probs(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the log probabilities for segmentation and classification.

        Args:
            data (dict): Input data for the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the segmentation probabilities and classification probabilities.
        """
        seg_probs, cls_probs = self.model.get_log_probs(data)
        return seg_probs, cls_probs
