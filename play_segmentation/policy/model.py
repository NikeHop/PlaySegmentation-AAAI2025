""" Imitation Learning NN policies """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from play_segmentation.utils.nn import (
    get_mlp,
    get_cov,
    GripperEncoderCalvin,
    ImageEncoderCalvin,
    ImageEncoderBabyAI,
    TrajectoryTransformer,
)


class MCILModel(nn.Module):
    """
    MCILModel is a PyTorch module that represents mulit-context imitation learning policy.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.

    Attributes:
        action_dim (int): The dimension of the action space.
        latent_dim (int): The dimension of the latent space.
        lang_dim (int): The dimension of the language input.
        n_dist (int): The number of mixture components for the action distribution.

        img_encoder (ImageEncoderCalvin): An instance of the ImageEncoderCalvin class for encoding RGB images.
        gripper_encoder (GripperEncoderCalvin): An instance of the GripperEncoderCalvin class for encoding gripper images.
        trajectory_encoder (TrajectoryTransformer): An instance of the TrajectoryTransformer class for encoding trajectories.
        encoder_output (nn.Linear): A linear layer for transforming the encoder output to the latent space.

        goal_emb_dim (int): The dimension of the goal embedding.
        lang_encoder (nn.Module): A multi-layer perceptron for encoding language inputs.
        state_encoder (nn.Module): A multi-layer perceptron for encoding state inputs.

        prior (nn.Module): A multi-layer perceptron for computing the prior distribution of latent variables.

        trajectory_decoder (TrajectoryTransformer): An instance of the TrajectoryTransformer class for decoding trajectories.
        decoder_means (nn.Linear): A linear layer for predicting the means of the action distribution.
        decoder_log_scales (nn.Linear): A linear layer for predicting the log scales of the action distribution.
        decoder_log_mixt_probs (nn.Linear): A linear layer for predicting the log mixture probabilities of the action distribution.

        decoder_binary_stop (nn.Linear): A linear layer for predicting the binary stop signal (if segmentation is enabled).

    Methods:
        forward(data): Performs the forward pass of the model.
        decode(data, latents): Decodes the latent variables into actions.
        prior_inference(data): Performs prior inference to generate actions.
        encode(data): Encodes the input data into latent variables.
        _create_decoder_seq(data, latents): Creates the input sequence for the decoder.
        _build_obs_encoder(data): Encodes the observation inputs.
        encode_goals(data): Encodes the goal inputs.
        sample_latents(mean, cov): Samples latent variables from a multivariate normal distribution.
        compute_prior(data): Computes the prior distribution of latent variables.
        get_prior(goal_emb): Computes the prior distribution of latent variables based on goal embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.action_dim = config["action_dim"]
        self.latent_dim = config["latent_dim"]
        self.lang_dim = config["lang_dim"]
        self.n_dist = config["n_dist"]

        # Perceptual encoders
        self.img_encoder = ImageEncoderCalvin(False)
        self.gripper_encoder = GripperEncoderCalvin(False)

        # Trajectory Encoder
        self.encoder_input_dim = config["obs_dim"] + config["gripper_obs_dim"]

        self.trajectory_encoder = TrajectoryTransformer(
            self.encoder_input_dim,
            config["encoder"]["input_dim"],
            config["encoder"]["nhead"],
            config["encoder"]["hidden_dim"],
            config["encoder"]["num_layers"],
            config["encoder"]["dropout"],
            True,
            "complete",
            config["encoder"]["context_length"],
            config["encoder"]["use_positional_encodings"],
        )

        self.encoder_output = nn.Linear(
            config["encoder"]["input_dim"], config["latent_dim"] * 2
        )

        # Goal Encoder
        self.goal_emb_dim = config["goal_encoder"]["output_dimension"]
        self.lang_encoder = get_mlp(
            self.lang_dim + self.encoder_input_dim,
            config["goal_encoder"]["hidden_dimension"],
            config["goal_encoder"]["output_dimension"],
            config["goal_encoder"]["n_layers"],
        )

        self.state_encoder = get_mlp(
            self.encoder_input_dim * 2,
            config["goal_encoder"]["hidden_dimension"],
            config["goal_encoder"]["output_dimension"],
            config["goal_encoder"]["n_layers"],
        )

        # Prior
        self.lang_dim = 384
        self.prior = get_mlp(
            config["goal_encoder"]["output_dimension"],
            config["prior"]["hidden_dim"],
            self.latent_dim * 2,
            config["prior"]["num_layers"],
        )

        # Trajectory Decoder
        decoder_input_dim = (
            config["latent_dim"]
            + self.encoder_input_dim
            + config["goal_encoder"]["output_dimension"]
        )

        self.trajectory_decoder = TrajectoryTransformer(
            decoder_input_dim,
            config["decoder"]["input_dim"],
            config["decoder"]["nhead"],
            config["decoder"]["hidden_dim"],
            config["decoder"]["num_layers"],
            config["decoder"]["dropout"],
            False,
            "forward",
            config["decoder"]["context_length"],
            config["decoder"]["use_positional_encodings"],
        )

        self.decoder_means = nn.Linear(
            config["decoder"]["input_dim"], self.action_dim * self.n_dist
        )
        self.decoder_log_scales = nn.Linear(
            config["decoder"]["input_dim"], self.action_dim * self.n_dist
        )
        self.decoder_log_mixt_probs = nn.Linear(
            config["decoder"]["input_dim"], self.action_dim * self.n_dist
        )

    def forward(
        self, data: dict
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass of the model.

        Args:
            data: Input data for the forward pass.

        Returns:
            actions: Decoded actions.
            variational_dist: Tuple containing the mean and covariance of the variational distribution.
            prior_dist: Tuple containing the mean and covariance of the prior distribution.
        """
        # Determine variational distribution
        variational_mean, variational_cov = self.encode(data)

        # Determine prior
        goal_emb = self.encode_goals(data)
        prior_mean, prior_cov = self.get_prior(goal_emb)

        # Prior
        latents = self.sample_latents(variational_mean, variational_cov)

        # Decoding
        actions = self.decode(data, latents)

        return actions, (variational_mean, variational_cov), (prior_mean, prior_cov)

    def decode(
        self, data: dict, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes the given data and latents into actions.

        Args:
            data (dict): The input data.
            latents (torch.Tensor): The latent tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The decoded actions.
        """
        trajectory = self._create_decoder_seq(data, latents)
        trajectory = self.trajectory_decoder(trajectory, data["obs_length"])
        means = self.decoder_means(trajectory)
        log_scales = self.decoder_log_scales(trajectory)
        log_mixt_probs = self.decoder_log_mixt_probs(trajectory)
        actions = (log_mixt_probs, means, log_scales)

        return actions

    def prior_inference(
        self, data: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform prior inference to generate actions based on the given data.

        Args:
            data: The input data used for prior inference.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The generated actions based on the prior inference.
        """
        self._build_obs_encoder(data)
        prior_mean, prior_cov = self.compute_prior(data)
        samples = self.sample_latents(prior_mean, prior_cov)
        actions = self.decode(data, samples)
        return actions

    def encode(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the given data into a latent representation.

        Args:
            data (dict): The input data containing trajectory information.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and covariance of the latent representation.
        """
        trajectory = self._build_obs_encoder(data)
        encoding = self.trajectory_encoder(trajectory, data["obs_length"])
        encoding = self.encoder_output(encoding)
        mean, cov = encoding[:, : self.latent_dim], F.softplus(
            encoding[:, self.latent_dim :]
        )
        cov = get_cov(cov)
        return mean, cov

    def _create_decoder_seq(self, data: dict, latents: torch.Tensor) -> torch.Tensor:
        """
        Create a decoder sequence by concatenating observations, latents, and goals.

        Args:
            data (dict): A dictionary containing the encoder observations and goals.
            latents (torch.Tensor): Latent tensor.

        Returns:
            torch.Tensor: The concatenated trajectory tensor.
        """
        _, T, _ = data["encoder_obs"].shape
        obs = data["encoder_obs"]
        latents = latents.unsqueeze(1).repeat(1, T, 1)
        goals = data["goals"].unsqueeze(1).repeat(1, T, 1)
        trajectory = torch.cat([obs, latents, goals], dim=-1)

        return trajectory

    def _build_obs_encoder(self, data: dict) -> torch.Tensor:
        """
        Builds the observation encoder for the given data.

        Args:
            data (dict): A dictionary containing the input data.

        Returns:
            torch.Tensor: The encoded observations.
        """
        # Encode RGB image
        B, T, H, W, C = data["img_obs"].shape
        img_encodings = self.img_encoder(
            data["img_obs"].reshape(B * T, W, H, C).permute(0, 3, 1, 2)
        ).reshape(B, T, -1)
        data["img_encodings"] = img_encodings

        # Encode Gripper Image
        B, T, H, W, C = data["gripper_obs"].shape
        gripper_encodings = self.gripper_encoder(
            data["gripper_obs"].reshape(B * T, W, H, C).permute(0, 3, 1, 2)
        ).reshape(B, T, -1)
        data["gripper_encodings"] = gripper_encodings

        data["encoder_obs"] = torch.cat([img_encodings, gripper_encodings], dim=-1)

        return data["encoder_obs"]

    def encode_goals(self, data: dict) -> torch.Tensor:
        """
        Encodes the goals in the given data.

        Args:
            data (dict): A dictionary containing the following keys:
                - "encoder_obs" (torch.Tensor): Tensor of shape (B, T, _) representing the encoder observations.
                - "mask" (torch.Tensor): Boolean tensor of shape (B,) indicating which goals to encode.

        Returns:
            torch.Tensor: Encoded goal embeddings of shape (B, self.goal_emb_dim).
        """
        B, T, _ = data["encoder_obs"].shape
        mask = data["mask"]
        goal_emb = torch.zeros(B, self.goal_emb_dim).to(data["encoder_obs"].device)

        # Embed goal images
        if not torch.all(mask):
            obs_length = data["obs_length"][~mask]
            start_states_goal = data["encoder_obs"][:, 0][~mask]
            goal_states = data["encoder_obs"][~mask]
            B_S, _, _ = goal_states.shape
            goal_states = goal_states[torch.arange(0, B_S), obs_length - 1]
            state_goals = self.state_encoder(
                torch.cat([start_states_goal, goal_states], dim=-1)
            )
            goal_emb[~mask] = state_goals

        # Embed instructions
        if torch.any(mask):
            start_states_lang = data["encoder_obs"][:, 0][mask]
            instructions = data["instructions"].float()[mask]
            lang_goals = self.lang_encoder(
                torch.cat([start_states_lang, instructions.squeeze(1)], dim=-1)
            )
            goal_emb[mask] = lang_goals

        data["goals"] = goal_emb

        return goal_emb

    def sample_latents(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Samples latent variables from a multivariate normal distribution.

        Args:
            mean (torch.Tensor): The mean of the multivariate normal distribution.
            cov (torch.Tensor): The covariance matrix of the multivariate normal distribution.

        Returns:
            torch.Tensor: Samples from the multivariate normal distribution.
        """
        mvn = MultivariateNormal(mean, cov)
        samples = mvn.rsample()
        return samples

    def compute_prior(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the prior distribution for the given data.

        Args:
            data (dict): A dictionary containing the input data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and covariance of the prior distribution.
        """
        if self.dataset == "calvin":
            data["instructions"] = data["instructions"].float()

        encoding = self.prior(
            torch.cat(
                [data["instructions"].squeeze(1), data["encoder_obs"][:, 0]], dim=-1
            )
        )
        mean, cov = encoding[:, : self.latent_dim], F.softplus(
            encoding[:, self.latent_dim :]
        )
        cov = get_cov(cov)
        return mean, cov

    def get_prior(self, goal_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the prior distribution parameters for the given goal embedding.

        Args:
            goal_emb (torch.Tensor): The goal embedding.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and covariance of the prior distribution.
        """
        latent_embeddings = self.prior(goal_emb)
        mean, cov = latent_embeddings[:, : self.latent_dim], F.softplus(
            latent_embeddings[:, self.latent_dim :]
        )
        cov = get_cov(cov)
        return mean, cov


class ILModel(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        Initializes an Imitation Learning policy for BabyAI.

        Args:
            config (dict): Configuration parameters for the model.

        """
        super().__init__()
        self.img_encoder = ImageEncoderBabyAI()

        input_dim = 128 * 8 * 8

        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, config["action_space"]),
        )

    def forward(self, obss: torch.Tensor, instructions: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            obss (torch.Tensor): Input observations.
            instructions (torch.Tensor): Input instructions.

        Returns:
            torch.Tensor: Predicted actions.

        """
        obss = self.img_encoder(obss, instructions)
        predicted_actions = self.policy(obss)
        return predicted_actions
