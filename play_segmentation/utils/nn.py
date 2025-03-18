"""Neural Networks Modules used to build models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


###### Functions ######


def get_mlp(
    input_dimension: int, hidden_dimension: int, output_dimension: int, n_layers: int
) -> nn.Module:
    """
    Create a multi-layer perceptron (MLP) neural network.

    Args:
        input_dimension (int): The dimension of the input features.
        hidden_dimension (int): The dimension of the hidden layers.
        output_dimension (int): The dimension of the output.
        n_layers (int): The number of hidden layers.

    Returns:
        nn.Module: The MLP neural network.

    Raises:
        AssertionError: If the number of layers is not a positive integer.
    """
    assert n_layers > 0, "Number of layers must be a positive integer"
    layers = [nn.Linear(input_dimension, hidden_dimension), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dimension, hidden_dimension), nn.ReLU()]
    layers.append(nn.Linear(hidden_dimension, output_dimension))
    return nn.Sequential(*layers)


def get_padding_mask(
    batch_size: int, length: int, seq_length: torch.Tensor, device
) -> torch.Tensor:
    """
    Generate a padding mask for sequences based on their lengths.

    Args:
        batch_size (int): The size of the batch.
        length (int): The maximum length of the sequences.
        seq_length (torch.tensor): A tensor containing the lengths of the sequences.
        device: The device to be used for the tensor operations.

    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, length) where each element is True if it corresponds to a padding position, and False otherwise.
    """
    padding_mask = torch.arange(0, length).expand(batch_size, length).to(
        device
    ) > seq_length.reshape(-1, 1)
    return padding_mask


def get_sequence_mask(
    T: int, context_length: int, obs_length: torch.Tensor, n_head: int, device
) -> torch.Tensor:
    """
    Get a mask that prevents the model from looking forward in time.

    Args:
        T (int): The length of the sequence.
        context_length (int): The length of the context window.
        obs_length (torch.Tensor): A tensor containing the lengths of the observed sequences.
        n_head (int): The number of attention heads.
        device: The device on which the mask should be created.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, T, T) that prevents the model from attending to future time steps.
    """
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(device)

    if context_length >= 0:
        mask = mask.unsqueeze(0).repeat(len(obs_length), 1, 1)
        for k, size in enumerate(obs_length):
            for i in range(size):
                if i > context_length:
                    mask[k, i, : i - context_length] = 1
        mask = torch.repeat_interleave(mask, n_head, dim=0)
    return mask


def get_cov(cov: torch.Tensor) -> torch.Tensor:
    """
    Transform a batch of vectors to diagonal covariance matrices.

    Args:
        cov (torch.Tensor): The input tensor representing the covariance matrix.

    Returns:
        torch.Tensor: The extended covariance matrix with diagonal elements.

    """
    B, D = cov.shape
    extended_cov = torch.zeros(B, D, D).to(cov.device)
    extended_cov[:, torch.arange(D), torch.arange(D)] = cov
    return extended_cov


def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log_sum_exp implementation that prevents overflow.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Result of the log_sum_exp operation.
    """
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


###### Neural Network Modules ######


class FocalLoss(nn.Module):
    def __init__(self, gamma: float):
        """
        Focal Loss constructor.

        Args:
            gamma (float): The focusing parameter gamma.
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Focal Loss.

        Args:
            logits (torch.Tensor): The predicted logits. Shape: BxC.
            gt (torch.Tensor): The ground truth labels. Shape: B.

        Returns:
            torch.Tensor: The computed focal loss.
        """
        prob = F.softmax(logits, dim=-1)
        prob = torch.gather(prob, dim=1, index=gt)
        return -((1 - prob) ** self.gamma) * prob.log()


class GripperEncoderCalvin(nn.Module):
    """
    GripperEncoderCalvin is a neural network module that encodes input data
    using convolutional and fully connected layers.

    Args:
        depth (int): Depth of the input image.
    """

    def __init__(self, depth: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3 + depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the output size of the last convolutional layer
        self.last_conv_output_size = 64 * 7 * 7

        self.fc1 = nn.Linear(self.last_conv_output_size, 256)
        self.fc2 = nn.Linear(256, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Spatial softmax
        x = F.softmax(x.view(x.size(0), x.size(1), -1), dim=-1)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ImageEncoderCalvin(nn.Module):
    def __init__(self, depth: int) -> None:
        """
        Image encoder module that encodes an input image into a feature vector.

        Args:
            depth (int): Depth of the input image.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3 + depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the output size of the last convolutional layer
        self.last_conv_output_size = 64 * 21 * 21

        self.fc1 = nn.Linear(self.last_conv_output_size, 512)
        self.fc2 = nn.Linear(512, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Spatial softmax
        x = F.softmax(x.view(x.size(0), x.size(1), -1), dim=-1)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        group_size=32,
        activation="relu",
    ):
        """
        A class representing a down-convolutional block in a neural network.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            padding (int, optional): The padding of the convolution. Defaults to 1.
            group_size (int, optional): The group size for group normalization. Defaults to 32.
            activation (str, optional): The activation function to use. Defaults to "relu".
        """
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.GroupNorm(group_size, out_channels)
        self.activation = get_activation(activation)
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the down-convolutional block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the down-convolutional block.
        """
        # Conv layers
        x = self.conv2d(x)
        # Normalization
        x = self.norm(x)
        # Activation
        x = self.activation(x)
        # Downsampling
        x = self.pooling(x)
        return x


class ImageEncoderBabyAI(nn.Module):
    """
    A class representing an image encoder for the BabyAI model.

    This class performs convolutional operations on input images and applies FiLM (Feature-wise Linear Modulation)
    to incorporate contextual information.

    Attributes:
        conv (nn.Sequential): A sequential module containing convolutional layers.
        film1 (FiLM): A FiLM module for the first FiLM operation.
        film2 (FiLM): A FiLM module for the second FiLM operation.
        pool (nn.MaxPool2d): A max pooling layer.

    Methods:
        forward(x, inst): Performs the forward pass of the image encoder.

    """

    def __init__(self):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ]

        self.conv = nn.Sequential(*layers)
        self.film1 = FiLM(
            in_features=512, out_features=128, in_channels=128, imm_channels=128
        )
        self.film2 = FiLM(
            in_features=512,
            out_features=128,
            in_channels=128,
            imm_channels=128,
        )

        self.pool = nn.MaxPool2d((7, 7), stride=2)

    def forward(self, x, inst):
        """
        Performs the forward pass of the image encoder.

        Args:
            x (torch.Tensor): Input image tensor.
            inst (torch.Tensor): Instruction tensor.

        Returns:
            torch.Tensor: Flattened output tensor.

        """
        x = self.conv(x)
        h = self.film1(x, inst)
        x = x + h
        h = self.film2(x, inst)
        x = x + h
        pool = False
        if pool:
            x = self.pool(x)
        return x.flatten(1, 3)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Positional encoding module for Transformer models.

        Args:
            d_model (int): The dimension of the input embeddings.
            dropout (float, optional): The dropout probability. Default is 0.1.
            max_len (int, optional): The maximum length of the input sequence. Default is 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding module.

        Args:
            x (torch.Tensor): The input tensor with shape [batch_size, seq_len, embedding_dim].

        Returns:
            torch.Tensor: The output tensor with shape [batch_size, seq_len, embedding_dim].
        """
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Args:
        num_channels (int): Number of input channels.
        eps (float, optional): Small value added to the denominator for numerical stability. Default is 1e-5.
        affine (bool, optional): If True, applies an affine transformation after normalization. Default is True.
        device (torch.device, optional): Device on which to allocate the tensors. If not specified, uses the default device.
        dtype (torch.dtype, optional): Data type of the tensors. If not specified, uses the default data type.

    Attributes:
        num_channels (int): Number of input channels.
        eps (float): Small value added to the denominator for numerical stability.
        affine (bool): If True, applies an affine transformation after normalization.
        weight (torch.nn.Parameter or None): Learnable weight parameter. If affine is False, set to None.
        bias (torch.nn.Parameter or None): Learnable bias parameter. If affine is False, set to None.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs)
            )
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: Normalized output tensor of the same shape as the input.
        """
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out = out * self.weight
            out = out + self.bias

        return out


class TrajectoryConvolution(nn.Module):
    """
    A convolutional neural network module for trajectory segmentation.

    Args:
        input_dim (int): The dimension of the input.
        d_dim (int): The dimension of the hidden state.
        context_length (int): The length of the context window.
        n_layers (int): The number of convolutional layers.

    Attributes:
        projection (nn.Linear): Linear layer for input projection.
        n_layers (int): The number of convolutional layers.
        convolutional_encoder (nn.ModuleList): List of convolutional layers.
        layer_norm (nn.ModuleList): List of layer normalization layers.
        relu (nn.ReLU): ReLU activation function.

    Methods:
        forward(seq, obs_length, inference=False): Forward pass of the module.
        _create_traj(seq): Create trajectory tensor from input sequence.
    """

    def __init__(
        self, input_dim: int, d_dim: int, context_length: int, n_layers: int
    ) -> None:
        super().__init__()

        self.projection = nn.Linear(input_dim, d_dim)

        self.n_layers = n_layers
        self.convolutional_encoder = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.relu = nn.ReLU()

        for layer in range(n_layers):
            layer = torch.nn.Conv1d(
                in_channels=d_dim,
                out_channels=d_dim,
                kernel_size=context_length,
                stride=1,
                padding=context_length - 1,
                padding_mode="zeros",
            )

            self.convolutional_encoder.append(layer)
            self.layer_norm.append(LayerNorm(d_dim))

    def forward(
        self, seq: dict, obs_length: int, inference: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the TrajectoryConvolution module.

        Args:
            seq (dict): Input sequence.
            obs_length (int): Length of the observed trajectory.
            inference (bool, optional): Flag for inference mode. Defaults to False.

        Returns:
            torch.Tensor: The output trajectory tensor.
        """
        trajectory = self._create_traj(seq)  # B,D,T

        for i in range(self.n_layers):
            _, _, T = trajectory.shape

            first_trajectory = self.convolutional_encoder[i](
                trajectory
            )  # B,D,T+context_length
            second_trajectory = first_trajectory[:, :, :T]
            third_trajectory = F.relu(self.layer_norm[i](second_trajectory))
            trajectory = third_trajectory.clone()

        return trajectory.permute(0, 2, 1)

    def _create_traj(self, seq: dict) -> torch.Tensor:
        """
        Create trajectory tensor from input sequence.

        Args:
            seq (dict): Input sequence.

        Returns:
            torch.Tensor: The trajectory tensor.
        """
        inputs = list(seq.values())
        return self.projection(torch.cat(inputs, dim=-1)).permute(0, 2, 1)


class TrajectoryTransformer(nn.Module):
    """
    A transformer-based model for trajectory processing.

    Args:
        input_dim (int): The dimension of the input trajectory.
        d_dim (int): The dimension of the transformer's hidden states.
        n_head (int): The number of attention heads in the transformer.
        hidden_dim (int): The dimension of the feedforward layer in the transformer.
        num_layers (int): The number of transformer layers.
        dropout (float): The dropout rate.
        aggregate (bool): Whether to aggregate the output trajectory.
        mask_type (str): The type of mask to apply during the transformer encoding.
        context_length (int): The length of the context window for the mask.
        use_positional_encoding (bool): Whether to use positional encoding in the transformer.

    Attributes:
        aggregate (bool): Whether to aggregate the output trajectory.
        mask_type (str): The type of mask to apply during the transformer encoding.
        use_positional_encoding (bool): Whether to use positional encoding in the transformer.
        context_length (int): The length of the context window for the mask.
        input_projection (nn.Linear): Linear layer for input projection.
        projection (nn.Linear): Linear layer for projection.
        positional_encoding (PositionalEncoding): Positional encoding layer.
        transformer_layer (nn.TransformerEncoderLayer): Transformer encoder layer.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder.
        n_head (int): The number of attention heads in the transformer.
    """

    def __init__(
        self,
        input_dim: int,
        d_dim: int,
        n_head: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        aggregate: bool,
        mask_type: str,
        context_length: int,
        use_positional_encoding: bool,
    ) -> None:
        super().__init__()

        self.aggregate = aggregate
        self.mask_type = mask_type
        self.use_positional_encoding = use_positional_encoding
        self.context_length = context_length
        self.input_projection = nn.Linear(input_dim, d_dim)
        self.projection = nn.Linear(input_dim, d_dim)
        self.positional_encoding = PositionalEncoding(d_dim, dropout=dropout)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        self.n_head = n_head

    def forward(
        self, trajectory: torch.Tensor, obs_length: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the TrajectoryTransformer.

        Args:
            trajectory (torch.Tensor): The input trajectory of shape (B, T, input_dim).
            obs_length (torch.Tensor): The observed length of each trajectory in the batch of shape (B,).

        Returns:
            torch.Tensor: The encoded trajectory of shape (B, T, d_dim) if aggregate is False,
                or the aggregated trajectory of shape (B, d_dim) if aggregate is True.

        """
        B, T, _ = trajectory.shape
        trajectory_mask = get_padding_mask(B, T, obs_length, trajectory.device)
        trajectory = self.input_projection(trajectory)

        if self.mask_type == "forward":
            key_mask = get_sequence_mask(
                T, self.context_length, obs_length, self.n_head, trajectory.device
            )

        elif self.mask_type == "complete":
            key_mask = torch.zeros(T, T).bool().to(trajectory.device)
        else:
            raise NotImplementedError("This mask type is not implemented")

        if self.use_positional_encoding:
            trajectory = self.positional_encoding(trajectory)

        trajectory = self.transformer_encoder(
            trajectory, mask=key_mask, src_key_padding_mask=trajectory_mask
        )

        if self.aggregate:
            trajectory[trajectory_mask] = 0
            trajectory = trajectory.sum(dim=1) / (
                T - trajectory_mask.sum(dim=1)
            ).reshape(-1, 1)

        return trajectory


def get_activation(fn_name: str) -> nn.Module:
    """
    Returns an activation function module based on the given function name.

    Args:
        fn_name (str): The name of the activation function.

    Returns:
        nn.Module: An instance of the activation function module.

    Raises:
        NotImplementedError: If the activation function is not implemented.
    """
    if fn_name == "relu":
        return nn.ReLU()
    else:
        raise NotImplementedError(f"Activation function {fn_name} not implemented")


# Adapted from BabyAI published by Maxime Chevalier-Boisvert, 2017:
# https://github.com/mila-iqia/babyai
# Licensed under the BSD-3-Clause License.


def initialize_parameters(m):
    """
    Initializes the parameters of a linear layer.

    Args:
        m (torch.nn.Module): The linear layer to initialize.

    Returns:
        None
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) module ().

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        in_channels (int): Number of input channels.
        imm_channels (int): Number of intermediate channels.

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer 1.
        bn1 (nn.BatchNorm2d): Batch normalization layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        bn2 (nn.BatchNorm2d): Batch normalization layer 2.
        weight (nn.Linear): Linear layer for weight modulation.
        bias (nn.Linear): Linear layer for bias modulation.

    Methods:
        forward(x, y): Forward pass of the FiLM module.

    """

    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=imm_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels,
            out_channels=out_features,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))
