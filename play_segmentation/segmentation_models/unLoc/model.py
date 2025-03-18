import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from play_segmentation.utils.nn import TrajectoryTransformer


class UnLocModel(nn.Module):
    """
    Reimplementation of the UnLoc model (https://arxiv.org/abs/2308.11062).

    Attributes:
        task (str): Task of the model.
        transformer (TrajectoryTransformer): Trajectory transformer of the model.
        temporal_convolutions (nn.ModuleList): Temporal convolutions of the model.
        relevance_head (Head): Relevance head of the model.
        boundary_head (Head): Boundary head of the model.

    Methods:
        __init__: Initializes the UnLocModel.
        forward: Performs forward pass of the UnLocModel.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the UnLocModel.

        Args:
            config (dict): Configuration parameters for the model.
        """
        super().__init__()

        self.task = config["task"]

        self.transformer = TrajectoryTransformer(
            config["transformer"]["input_dim"],
            config["transformer"]["input_dim"],
            config["transformer"]["n_head"],
            config["transformer"]["hidden_dim"],
            config["transformer"]["num_layers"],
            config["transformer"]["dropout"],
            False,
            "complete",
            config["transformer"]["context_length"],
            config["transformer"]["use_positional_encodings"],
        )

        self.temporal_convolutions = nn.ModuleList()

        for kernel_size in config["temporal_convolution"]["kernel_sizes"]:
            layer = nn.Conv1d(
                config["transformer"]["input_dim"],
                config["transformer"]["input_dim"],
                kernel_size=kernel_size,
                stride=2,
            )
            self.temporal_convolutions.append(layer)

        self.relevance_head = Head(
            output_dim=1,
            n_layers=config["head"]["n_layers"],
            n_channels=config["transformer"]["input_dim"],
            kernel_size=config["head"]["kernel_size"],
        )
        self.boundary_head = Head(
            output_dim=2,
            n_layers=config["head"]["n_layers"],
            n_channels=config["transformer"]["input_dim"],
            kernel_size=config["head"]["kernel_size"],
        )

    def forward(self, data: dict) -> dict:
        """
        Performs forward pass of the UnLocModel.

        Args:
            data (dict): Input data for the model.

        Returns:
            dict: Predictions of the model.

        """
        B, C, T, D = data["clip_seq"].shape
        BC = B * C

        seq = rearrange(data["clip_seq"], "b c t d -> (b c) t d")

        # Apply Transformer
        obs_length = data["obs_lengths"].repeat_interleave(C)
        seq = self.transformer(seq, obs_length)  # BC x T+S x D

        # Separate image frames from lang frames
        img_seqs = []
        for i, l in enumerate(data["img_obs_lengths"]):
            img_seqs.append(rearrange(seq[i * C : (i + 1) * C, :l], "c t d -> t c d"))
        seq = pad_sequence(img_seqs, batch_first=True)  # BxTxCxD
        seq = rearrange(seq, "b t c d -> (b c) t d")

        # Apply temporal convolutions for FP (only if not action segmentation)
        feature_pyramid = [seq]
        obs_lengths = [seq.shape[1]]
        if self.task != "action_segmentation":
            for layer in self.temporal_convolutions:
                seq = rearrange(seq, "b t c -> b c t")
                seq = layer(seq)
                seq = rearrange(seq, "b c t -> b t c")

                t = seq.shape[1]
                seq_alias = torch.zeros(BC, T, D)
                seq_alias[:, :t, :] = seq.copy()
                seq = seq_alias

                feature_pyramid.append(seq)
                obs_lengths.append(t)

        # Apply Heads
        features = torch.cat(feature_pyramid, dim=0)
        relevance_pred = self.relevance_head(features)
        boundary_pred = F.relu(self.boundary_head(features))

        # Parse the output by the layers
        for i in range(len(feature_pyramid)):
            relevance_score = relevance_pred[
                BC * i : BC * (i + 1), : obs_lengths[i]
            ]  # BC x T_n x D
            boundary_score = boundary_pred[
                BC * i : BC * (i + 1) :, : obs_lengths[i]
            ]  # BC x T_n x D
            relevance_score = rearrange(relevance_score, "(b c) t d -> b c t d", b=B)
            boundary_score = rearrange(boundary_score, "(b c) t d -> b c t d", b=B)
            predictions = {f"layer_{i}": (relevance_score, boundary_score)}

        return predictions


class Head(nn.Module):
    def __init__(
        self, output_dim: int, n_layers: int, n_channels: int, kernel_size: int
    ):
        """
        Head module for the segmentation model.

        Args:
            output_dim (int): The dimension of the output.
            n_layers (int): The number of layers in the head.
            n_channels (int): The number of channels in the head.
            kernel_size (int): The kernel size for the head layers.
        """
        super().__init__()

        layers = []
        for _ in range(n_layers):
            l = HeadLayer(n_channels, kernel_size)
            layers.append(l)

        self.layers = nn.Sequential(*layers)
        self.projection = nn.Linear(n_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size, T is the sequence length, and C is the number of channels.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B = x.shape[0]
        x = self.layers(x)
        x = rearrange(x, "b t c -> (b t) c")
        x = self.projection(x)
        x = rearrange(x, "(b t) c -> b t c", b=B)
        return x


class HeadLayer(nn.Module):
    def __init__(self, n_channels: int, kernel_size: int):
        """
        HeadLayer class represents a head layer in a neural network model.

        Args:
            n_channels (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()

        self.ln = nn.LayerNorm(n_channels)
        self.convolution = nn.Conv1d(
            n_channels, n_channels, kernel_size=kernel_size, stride=1
        )
        self.relu = nn.ReLU()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.ln(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.convolution(
            F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2))
        )
        x = rearrange(x, "b c t -> b t c")
        x = self.relu(x)
        return x
