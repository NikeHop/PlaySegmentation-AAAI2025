from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from play_segmentation.labelling.labelling_training.trainer import I3D
from play_segmentation.utils.nn import (
    TrajectoryConvolution,
    TrajectoryTransformer,
)


class PlaySegmentationModel(nn.Module):
    """
    PlaySegmentationModel is a PyTorch module for trajectory segmentation.

    Args:
        config (dict): Configuration parameters for the model.

    Attributes:
        timestep_loss (bool): Flag indicating whether to calculate per timestep loss.
        use_language (bool): Flag indicating whether to use language in the model.
        trajectory_embedding (int): Dimension of the trajectory embedding.
        architecture (str): Model architecture type.
        i3d_features (bool): Flag indicating whether to use I3D features.
        obs_encoder (nn.Module): Encoder module for observation data.
        encoder (nn.Module): Encoder module for trajectory data.
        linear (nn.Linear): Linear layer for segment probability.
        cls_linear (nn.Linear): Linear layer for classification output.

    Methods:
        forward(data): Performs a forward pass through the model.
        encode_obs(data): Encodes observation data.
        encode_trajectory(data): Encodes trajectory data.
        get_log_probs(data): Calculates log probabilities for stop and class labels.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.use_language = config["use_language"]
        if self.use_language:
            self.trajectory_embedding = config["obs_dim"] + 384
        else:
            self.trajectory_embedding = config["obs_dim"]

        self.obs_encoder = I3D.load_from_checkpoint(config["i3d"]["checkpoint"])

        self.architecture = config["architecture"]
        if self.architecture == "transformer":
            self.encoder = TrajectoryTransformer(
                self.trajectory_embedding,
                config["transformer"]["d_dim"],
                config["transformer"]["n_head"],
                config["transformer"]["hidden_dim"],
                config["transformer"]["num_layers"],
                config["transformer"]["dropout"],
                False,
                "forward",
                config["transformer"]["context_length"],
                config["transformer"]["use_positional_encoding"],
            )
            output_dim = config["transformer"]["d_dim"]
        elif self.architecture == "convolution":
            self.encoder = TrajectoryConvolution(
                self.trajectory_embedding,
                512,
                config["conv"]["context_length"],
                config["conv"]["n_layers"],
            )
            output_dim = 512
        else:
            raise NotImplementedError("This model architecture does not exist")

        self.linear = nn.Linear(output_dim, 1)
        self.cls_linear = nn.Linear(output_dim, config["num_classes"])

    def forward(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the model.

        Args:
            data (dict): Input data for the forward pass.

        Returns:
            tuple: A tuple containing the regression probabilities and classification probabilities.
        """
        self.encode_obs(data)
        trajectory = self.encode_trajectory(data)  # (B,T,D)
        B = trajectory.shape[0]
        trajectory = trajectory[torch.arange(0, B), data["obs_length"] - 1]
        prob = torch.sigmoid(self.linear(trajectory))  # Bx1

        cls_mask = data["class_labels"] > -1
        cls_prob = self.cls_linear(trajectory[cls_mask])

        return prob.squeeze(-1), cls_prob

    def encode_obs(self, data: dict) -> None:
        """
        Encodes observation data.

        Args:
            data (dict): Input data containing observation images.

        Returns:
            None
        """
        with torch.no_grad():
            data["obs"] = self.obs_encoder.encode(data["img_obs"])
        data["obs"] = rearrange(data["obs"], "b c t -> b t c")

    def encode_trajectory(self, data: dict) -> torch.Tensor:
        """
        Encodes trajectory data.

        Args:
            data (dict): Input data containing trajectory observations.

        Returns:
            torch.Tensor: Encoded trajectory tensor.
        """
        # Trajectory BxTxD
        if self.architecture == "transformer":
            trajectory = self.encoder(data["obs"], data["obs_length"])
        else:
            seq = {"obs": data["obs"]}
            trajectory = self.encoder(seq, data["obs_length"])

        return trajectory

    def get_log_probs(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates log probabilities for stop and class labels.

        Args:
            data (dict): Input data for calculating log probabilities.

        Returns:
            tuple: A tuple containing the log probabilities for stop and class labels.
        """
        self.encode_obs(data)
        trajectory = self.encode_trajectory(data)  # (B,T,D)
        stop_log_prob = F.logsigmoid(self.linear(trajectory)).squeeze(-1)
        cls_log_prob = F.log_softmax(self.cls_linear(trajectory), dim=-1)
        return stop_log_prob, cls_log_prob
