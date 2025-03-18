"""PyTorch Lightning trainer for the MCIL policy."""

from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import Adam
from transformers import AutoTokenizer, T5EncoderModel

from play_segmentation.policy.model import MCILModel, ILModel
from play_segmentation.utils.nn import get_padding_mask, log_sum_exp


class MCIL(pl.LightningModule):
    """
    Multi-Context Imitation Learning (MCIL) LightningModule.

    This class represents the MCIL model for training and evaluation.
    It extends the `pl.LightningModule` class from PyTorch Lightning.

    Args:
        config (dict): Configuration parameters for the MCIL model.

    Attributes:
        model (MCILModel): The MCIL model.
        lr (float): Learning rate for the optimizer.
        loss (nn.CrossEntropyLoss): Cross-entropy loss function.
        binary_weight (float): Weight for the binary stop loss term.
        beta (float): Weight for the KL divergence term.
        std (float): Standard deviation for the Mixture of Logistics distribution.
        action_dim (int): Dimension of the action space.
        log_scale_min (float): Minimum value for the log scale.
        n_dist (int): Number of distributions in the Mixture of Logistics.
        num_classes (int): Number of classes in the action space.
        segmentation (bool): Flag indicating whether to perform segmentation.
        only_stop (bool): Flag indicating whether to only consider stop actions.
        action_min_bound (torch.Tensor): Minimum bound for the action space.
        action_max_bound (torch.Tensor): Maximum bound for the action space.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MCILModel(config["model"])

        self.lr = config["lr"]
        self.loss = nn.CrossEntropyLoss(reduction="none")

        self.beta = config["beta"]
        self.std = config["std"]

        self.action_dim = config["model"]["action_dim"]
        self.log_scale_min = config["log_scale_min"]
        self.n_dist = config["model"]["n_dist"]
        self.num_classes = config["num_classes"]
        self.segmentation = config["model"]["segmentation"]
        self.only_stop = config["model"]["only_stop"]

        self.register_buffer(
            "action_min_bound",
            (
                -torch.ones(self.action_dim)
                .to(self.device)
                .reshape(1, 1, len(config["act_min_bound"]))
                .unsqueeze(-1)
                .repeat(1, 1, 1, self.n_dist)
            ),
        )

        self.register_buffer(
            "action_max_bound",
            (
                torch.ones(self.action_dim)
                .to(self.device)
                .reshape(1, 1, len(config["act_min_bound"]))
                .unsqueeze(-1)
                .repeat(1, 1, 1, self.n_dist)
            ),
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Adam: The Adam optimizer with the model parameters and learning rate.
        """
        return Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, data: dict, data_id: int) -> float:
        """
        Perform a single training step.

        Args:
            data: The input data for the training step.
            data_id: The ID of the data.

        Returns:
            The loss value for the training step.
        """
        actions, variational_param, prior_param = self.model(data)
        nll, kl = self.compute_elbo(actions, variational_param, prior_param, data)
        loss = nll + self.beta * kl

        info: dict[str, float] = {
            "loss": loss,
            "nll": nll,
            "kl": kl,
        }
        metrics = self.get_metrics(actions, data)
        info.update(metrics)

        for key, value in info.items():
            self.log(f"training/{key}", value)

        return loss

    def validation_step(self, data: dict, data_id: int) -> None:
        """
        Perform a validation step for the policy trainer.

        Args:
            data (dict): The input data for the validation step.
            data_id (int): The ID of the data.

        Returns:
            None
        """
        actions, variational_param, prior_param = self.model(data)
        actions = self.model.decode(data, prior_param[0])
        metrics = self.get_metrics(actions, data)

        for key, value in metrics.items():
            self.log(f"validation/{key}", value, sync_dist=True)

    def get_metrics(self, actions: torch.Tensor, data: dict) -> dict:
        """
        Compute various metrics based on the predicted actions and the ground truth actions.

        Args:
            actions (torch.Tensor): Predicted actions of shape (B, T, C), where B is the batch size,
                T is the sequence length, and C is the number of action dimensions.
            data (dict): Dictionary containing the ground truth actions and other relevant data.

        Returns:
            dict: A dictionary containing the computed metrics:
                - "action_accuracy": The mean absolute difference between the predicted actions and the ground truth actions.
                - "gripper_accuracy": The accuracy of the gripper actions.
                - "lang_action_accuracy": The mean absolute difference between the predicted actions and the ground truth actions,
                  considering only the data points with non-zero language mask.
                - "lang_gripper_accuracy": The accuracy of the gripper actions, considering only the data points with non-zero language mask.
                - "goal_action_accuracy": The mean absolute difference between the predicted actions and the ground truth actions,
                  considering only the data points with zero language mask.
                - "goal_gripper_accuracy": The accuracy of the gripper actions, considering only the data points with zero language mask.
        """
        # Sample an action
        actions = self._sample_actions(actions)

        # Compute absolute difference
        B, T, _ = actions.shape
        mask = ~get_padding_mask(B, T, data["obs_length"], self.device)

        abs_diff = torch.abs(actions[:, :, :-1] - data["actions"][:, :, :-1])[
            mask
        ].mean()

        # Gripper accuracy
        pos_gripper_mask = torch.logical_and(
            data["actions"][:, :, 6] > 0, actions[:, :, 6] > 0
        )
        neg_gripper_mask = torch.logical_and(
            data["actions"][:, :, 6] < 0, actions[:, :, 6] < 0
        )
        gripper_accuracy = (pos_gripper_mask + neg_gripper_mask)[mask].float().mean()

        # Compute based on language and goal
        if torch.any(data["mask"]):
            lang_padding_mask = mask[data["mask"]]
            lang_action_accuracy = torch.abs(
                actions[:, :, :-1] - data["actions"][:, :, :-1]
            )[data["mask"]][lang_padding_mask].mean()
            lang_gripper_accuracy = (
                (pos_gripper_mask + neg_gripper_mask)[data["mask"]][lang_padding_mask]
                .float()
                .mean()
            )
        else:
            lang_action_accuracy = 0
            lang_gripper_accuracy = 0

        if not torch.all(data["mask"]):
            goal_padding_mask = mask[~data["mask"]]
            goal_action_accuracy = torch.abs(
                actions[:, :, :-1] - data["actions"][:, :, :-1]
            )[~data["mask"]][goal_padding_mask].mean()
            goal_gripper_accuracy = (
                (pos_gripper_mask + neg_gripper_mask)[~data["mask"]][goal_padding_mask]
                .float()
                .mean()
            )
        else:
            goal_action_accuracy = 0
            goal_gripper_accuracy = 0

        return {
            "action_accuracy": abs_diff,
            "gripper_accuracy": gripper_accuracy,
            "lang_action_accuracy": lang_action_accuracy,
            "lang_gripper_accuracy": lang_gripper_accuracy,
            "goal_action_accuracy": goal_action_accuracy,
            "goal_gripper_accuracy": goal_gripper_accuracy,
        }

    def compute_kl(
        self,
        variational: tuple[torch.Tensor, torch.Tensor],
        prior: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """
        Compute the Kullback-Leibler (KL) divergence between two multivariate normal distributions.

        Parameters:
            variational (tuple): A tuple containing the mean and covariance matrix of the variational distribution.
            prior (tuple): A tuple containing the mean and covariance matrix of the prior distribution.

        Returns:
            kl (float): The KL divergence between the variational and prior distributions.
        """
        variational_mvn = MultivariateNormal(variational[0], variational[1])
        prior_mvn = MultivariateNormal(prior[0], prior[1])
        kl = kl_divergence(variational_mvn, prior_mvn)
        return kl

    def compute_elbo(
        self,
        actions: list,
        variational: torch.Tensor,
        prior_param: torch.Tensor,
        data: dict,
    ) -> tuple:
        """
        Computes the evidence lower bound (ELBO) for the given inputs.

        Args:
            actions (list): A list of tensors representing the actions.
            variational (Tensor): The variational parameters.
            prior_param (Tensor): The prior parameters.
            data (dict): A dictionary containing the data.

        Returns:
            tuple: A tuple containing the negative log-likelihood (nll) and the Kullback-Leibler divergence (kl).
        """
        B, T, _ = actions[0].shape
        mask = ~get_padding_mask(B, T, data["obs_length"], self.device)
        nll = self.get_loss(actions, data, mask)
        kl = self.compute_kl(variational, prior_param).mean()

        return nll, kl

    def get_loss(
        self, actions: torch.Tensor, data: dict, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the action loss for the given actions, data, and mask.

        Args:
            actions (torch.Tensor): The actions predicted by the model.
            data (dict): The input data containing the ground truth actions.
            mask (torch.Tensor): The mask indicating valid actions.

        Returns:
            torch.Tensor: The action loss.

        """
        # Parameters of the Mixture of logisitics distribution
        log_mixt_probs, means, log_scales = actions

        B, T, _ = means.shape
        log_mixt_probs = log_mixt_probs.reshape(B, T, self.action_dim, self.n_dist)
        means = means.reshape(B, T, self.action_dim, self.n_dist)
        log_scales = log_scales.reshape(B, T, self.action_dim, self.n_dist)

        # Clamp scale
        log_scales = torch.clamp(log_scales, min=self.log_scale_min)

        # Copy actions for mixture
        actions = data["actions"].unsqueeze(-1).repeat(1, 1, 1, self.n_dist)

        centered_actions = actions - means
        inv_stdv = torch.exp(-log_scales)
        act_range = (self.action_max_bound - self.action_min_bound) / 2.0
        plus_in = inv_stdv * (centered_actions + act_range / (self.num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_actions - act_range / (self.num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # Corner Cases
        log_cdf_plus = plus_in - F.softplus(
            plus_in
        )  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(
            min_in
        )  # log probability for edge case of 255 (before scaling)
        # Log probability in the center of the bin
        mid_in = inv_stdv * centered_actions
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        # Probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        # Log probability
        log_probs = torch.where(
            actions < self.action_min_bound + 1e-3,
            log_cdf_plus,
            torch.where(
                actions > self.action_max_bound - 1e-3,
                log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                    log_pdf_mid - np.log((self.num_classes - 1) / 2),
                ),
            ),
        )

        log_probs = log_probs + F.log_softmax(log_mixt_probs, dim=-1)
        action_loss = torch.sum(log_sum_exp(log_probs), dim=-1).float()

        action_loss = -action_loss[mask].mean()

        return action_loss

    def _sample_actions(self, actions: tuple) -> torch.Tensor:
        """
        Sample actions from the given action distribution.

        Args:
            actions (tuple): A tuple containing logit_mixt_probs, means, and log_scales.

        Returns:
            torch.Tensor: Sampled actions.
        """
        logit_mixt_probs, means, log_scales = actions

        B, T, _ = log_scales.shape
        log_scales = log_scales.reshape(B, T, self.action_dim, self.n_dist)
        means = means.reshape(B, T, self.action_dim, self.n_dist)
        logit_mixt_probs = logit_mixt_probs.reshape(B, T, self.action_dim, self.n_dist)

        r1, r2 = 1e-5, 1.0 - 1e-5
        temp = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        temp = logit_mixt_probs - torch.log(-torch.log(temp))
        argmax = torch.argmax(temp, -1)

        dist = torch.eye(self.n_dist, device=self.device)[argmax]

        # Select scales and means
        log_scales = (dist * log_scales).sum(dim=-1)
        means = (dist * means).sum(dim=-1)

        # Inversion sampling for logistic mixture sampling
        scales = torch.exp(log_scales)  # Make positive
        u = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        actions = means + scales * (torch.log(u) - torch.log(1.0 - u))

        return actions

    def act(
        self, data: torch.Tensor, device: torch.device, resample: bool = False
    ) -> torch.Tensor:
        """
        Generates actions based on the given data.

        Args:
            data (torch.Tensor): Input data for generating actions.
            device (torch.device): Device to perform the computation on.
            resample (bool, optional): Whether to resample actions. Defaults to False.

        Returns:
            torch.Tensor: Generated actions.
        """
        self.to_device(data, device)
        actions, variational_param, prior_param = self.model(data)

        if resample:
            self.last_prior_param = prior_param

        prior_param = self.last_prior_param

        actions = self.model.decode(data, prior_param[0])

        actions = self._sample_actions(actions)
        actions = actions[0][-1].reshape(1, -1)
        return actions

    def act_with_teacher(self, data: dict) -> torch.Tensor:
        """
        Generates actions based on the given data using the teacher model.

        Args:
            data (dict): The input data for generating actions.

        Returns:
            torch.Tensor: The generated actions.
        """
        self.to_device(data)
        actions, variational_param, prior_param = self.model(data)

        if self.data_parallel:
            actions = self.model.module.decode(data, prior_param[0])
        else:
            actions = self.model.decode(data, prior_param[0])

        actions = self._sample_actions(actions)
        actions[:, :, -1] = torch.where(actions[:, :, -1] > 0, 1, -1)
        return actions

    def to_device(self, data: dict, device: torch.device) -> dict:
        """
        Move the tensors in the given data dictionary to the specified device.

        Args:
            data (dict): A dictionary containing tensors to be moved to the device.
            device (torch.device): The target device to move the tensors to.

        Returns:
            dict: A dictionary with tensors moved to the specified device.
        """
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
        return data


class IL(pl.LightningModule):
    """
    Class representing an imitation learning (IL) model.

    Args:
        config (dict): Configuration parameters for the IL model.

    Attributes:
        model (ILModel): The IL model.
        loss (nn.CrossEntropyLoss): The loss function for training.
        lr (float): The learning rate for the optimizer.

    Methods:
        configure_optimizers: Configures the optimizer for training.
        training_step: Performs a single training step.
        compute_loss: Computes the loss for the given predictions and ground truth.
        compute_metrics: Computes the evaluation metrics.
        validation_step: Performs a single validation step.
        load_instruction_embeddings: Loads the instruction embeddings.
        embed_mission: Embeds a mission using the instruction embeddings.
        format_model_input: Formats the model input.
        act: Performs an action based on the given observations and observation type.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = ILModel(config["model"])
        self.loss = nn.CrossEntropyLoss(reduction="none")

        # Hyperparameters
        self.lr = config["lr"]

    def configure_optimizers(self) -> Adam:
        """
        Configures the optimizer for the model.

        Returns:
            Adam: The optimizer for the model.
        """
        return Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, data: dict[str, Any], data_id: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            data (Dict[str, Any]): The input data for the training step.
            data_id (int): The ID of the data.

        Returns:
            torch.Tensor: The loss value for the training step.
        """
        predictions = self.model(data["obss"], data["instructions"])
        loss = self.compute_loss(predictions, data)
        self.log("training/loss", loss.detach())
        info = self.compute_metrics(predictions, data)
        for key, value in info.items():
            self.log(f"training/{key}", value)
        return loss

    def compute_loss(
        self, predictions: torch.Tensor, data: dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute the loss for the given predictions and data.

        Args:
            predictions (Any): The predicted actions.
            data (Dict[str, Any]): The input data containing the ground truth actions.

        Returns:
            float: The computed loss.

        """
        gt = data["actions"]
        loss = self.loss(predictions, gt).mean()
        return loss

    def compute_metrics(
        self, predictions: torch.Tensor, data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Compute the accuracy metric for the given predictions and ground truth data.

        Args:
            predictions (torch.Tensor): The predicted actions.
            data (Dict[str, torch.Tensor]): The ground truth data.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the accuracy metric.
        """
        with torch.no_grad():
            predicted_actions = predictions.argmax(dim=-1)
            gt_actions = data["actions"]
            acc = (predicted_actions == gt_actions).float().mean()
        return {"accuracy": acc}

    def validation_step(self, data: dict, data_id: int) -> None:
        """
        Perform a validation step for the policy trainer.

        Args:
            data (dict): The input data for the validation step.
            data_id (int): The ID of the data.

        Returns:
            None
        """
        predictions = self.model(data["obss"], data["instructions"])
        loss = self.compute_loss(predictions, data)
        self.log("validation/loss", loss.detach())
        info = self.compute_metrics(predictions, data)
        for key, value in info.items():
            self.log(f"validation/{key}", value)

    def load_instruction_embeddings(self, embedding_config: dict) -> None:
        """
        Load instruction embeddings from the specified configuration.

        Args:
            embedding_config (dict): A dictionary containing the configuration for the embeddings.
                It should have the following keys:
                - "path" (str): The path to the saved instruction embeddings.
                - "tokenizer" (str): The name or path of the tokenizer used for the embeddings.
                - "model" (str): The name or path of the model used for the embeddings.

        Returns:
            None
        """
        self.instruction_embeddings = torch.load(open(embedding_config["path"], "rb"))
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            embedding_config["tokenizer"]
        )
        self.embedding_model = T5EncoderModel.from_pretrained(embedding_config["model"])

    def embed_mission(self, mission: str) -> torch.Tensor:
        """
        Embeds the given mission using the instruction embeddings model.

        Args:
            mission (str): The mission to be embedded.

        Returns:
            torch.Tensor: The embedded representation of the mission.
        """
        if mission in self.instruction_embeddings:
            return self.instruction_embeddings[mission]

        model_input = self.embedding_tokenizer(
            [mission], truncation=True, return_tensors="pt"
        )
        embedding = self.embedding_model(**model_input).last_hidden_state.mean(dim=1)
        self.instruction_embeddings[mission] = embedding
        return embedding

    def format_model_input(
        self, obss: list[dict[str, Any]], obs_type: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Formats the model input by embedding the mission and rearranging the observations.

        Args:
            obss (List[Dict[str, Any]]): List of observation dictionaries.
            obs_type (str): Type of observation to use.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the formatted observations and the embedded mission.
        """
        mission = torch.cat(
            [
                self.embed_mission(obs["mission"]).reshape(1, -1).to(self.device)
                for obs in obss
            ],
            dim=0,
        )
        obs = torch.stack(
            [torch.tensor(obs[obs_type], dtype=torch.float) for obs in obss], dim=0
        ).to(self.device)
        obs = rearrange(obs, "b h w c -> b c h w")

        return (obs, mission)

    def act(self, obs: list[dict[str, Any]], obs_type: str) -> list[int]:
        """
        Takes an observation and its type and returns the corresponding actions.

        Args:
            obs (list[dict[str, Any]]): The observation.
            obs_type (str): The type of the observation.

        Returns:
            actions (list[int]): The actions corresponding to the given observation.

        """
        model_input = self.format_model_input(obs, obs_type)
        with torch.no_grad():
            actions = self.model(*model_input).argmax(dim=-1)
            actions = actions.tolist()
        return actions
