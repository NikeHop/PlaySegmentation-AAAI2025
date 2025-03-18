""" Trainer class for the UnLoc model. """

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.optim import Adam

from play_segmentation.segmentation_models.unLoc.model import UnLocModel
from play_segmentation.utils.nn import FocalLoss


class UnLoc(pl.LightningModule):
    def __init__(self, config):
        """
        Initializes the UnLoc model.

        Args:
            config (dict): Configuration parameters for the model.
        """
        super().__init__()

        self.save_hyperparameters()

        self.lr = config["lr"]
        self.task = config["model"]["task"]
        self.model = UnLocModel(config["model"])

        self.cross_entropy = nn.CrossEntropyLoss(reduce="mean")
        self.focal_loss = FocalLoss(config["gamma"])
        self.l1 = nn.L1Loss(reduce="mean")

    def training_step(self, data: dict, idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            data (dict): Input data for the training step.
            idx (int): Index of the training step.

        Returns:
            torch.Tensor: Loss value for the training step.
        """
        predictions = self.model(data)

        if self.task == "action_segmentation":
            relevance_scores = predictions["layer_0"][0]
            relevance_scores = rearrange(
                relevance_scores, "b c t d -> (b t) c d"
            ).squeeze(-1)

            gt_classes = rearrange(data["classes"], "b t -> (b t)")
            gt_classes = gt_classes[data["mask"]]
            relevance_scores = relevance_scores[data["mask"]]
            loss = self.cross_entropy(relevance_scores, gt_classes)
            predicted_classes = torch.argmax(relevance_scores, dim=-1)
            acc = (predicted_classes == gt_classes).float().mean()

        elif self.task == "action_localization":
            raise NotImplementedError(
                f"This action type is not implemented yet {self.task}"
            )

        else:
            raise NotImplementedError(
                f"This action type has not been implemented {self.task}"
            )

        self.log("training/loss", loss.detach())
        self.log("training/frame_level_accuracy", acc.detach())

        return loss

    def validation_step(self, data: dict, idx: int) -> None:
        """
        Performs a single validation step.

        Args:
            data (dict): Input data for the validation step.
            idx (int): Index of the validation step.
        """
        predictions = self.model(data)

        if self.task == "action_segmentation":
            relevance_scores = predictions["layer_0"][0]

            relevance_scores = rearrange(
                relevance_scores, "b c t d-> (b t) c d"
            ).squeeze(-1)
            gt_classes = rearrange(data["classes"], "b t -> (b t)")
            gt_classes = gt_classes[data["mask"]]
            relevance_scores = relevance_scores[data["mask"]]

            predicted_classes = torch.argmax(relevance_scores, dim=-1)
            acc = (predicted_classes == gt_classes).float().mean()
        else:
            raise NotImplementedError(
                f"This action type has not been implemented {self.task}"
            )

        self.log("validation/frame_level_accuracy", acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        return Adam(self.model.parameters(), lr=self.lr)

    def get_log_probs(self, data: dict, device: torch.device) -> torch.Tensor:
        """
        Calculates the log probabilities for the given data.

        Args:
            data (dict): Input data for calculating log probabilities.
            device (torch.device): The device to be used for calculations.

        Returns:
            torch.Tensor: Log probabilities for the given data.
        """
        batch: dict = {}

        C: int = data["instructions"].shape[0]
        T: int = data["clip_img_seq"].shape[0]

        clip_seq: torch.Tensor = torch.cat(
            [
                data["clip_img_seq"].unsqueeze(1).repeat(1, C, 1),
                data["instructions"].unsqueeze(0),
            ],
            dim=0,
        )

        batch["clip_seq"] = (
            rearrange(clip_seq, "t c d -> c t d").unsqueeze(0).to(device)
        )
        batch["obs_lengths"] = torch.tensor([clip_seq.shape[0]]).long().to(device)
        batch["img_obs_lengths"] = torch.tensor([T]).long().to(device)

        predictions = self.model(batch)

        relevance_scores: torch.Tensor = predictions["layer_0"][0]
        relevance_scores = rearrange(relevance_scores, "b c t d-> b t c d").squeeze(-1)
        log_probs: torch.Tensor = F.log_softmax(relevance_scores, dim=-1)

        return log_probs

    def segment_segment(self, data: dict) -> torch.Tensor:
        """
        Segments the given data.

        Args:
            data (dict): Input data to be segmented.

        Returns:
            torch.Tensor: Probability of the instruction classes at each timestep.
        """
        predictions = self.model(data)
        relevance_scores = predictions["layer_0"][0]
        relevance_scores = rearrange(relevance_scores, "b c t d-> b t c d").squeeze(-1)
        probs = F.softmax(relevance_scores, dim=-1)

        return probs
