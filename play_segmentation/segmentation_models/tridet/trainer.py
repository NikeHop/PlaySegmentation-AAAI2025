import lightning.pytorch as pl
import numpy as np
import torch
from torch.optim import Adam

from play_segmentation.labelling.labelling_training.trainer import I3D
from play_segmentation.segmentation_models.tridet.model import TriDetModel


class TriDet(pl.LightningModule):
    """
    TriDet is a class that represents the LightningModule for training and evaluating the TriDet model.

    Args:
        config (dict): Configuration dictionary containing the model and training parameters.

    Attributes:
        lr (float): Learning rate for the optimizer.
        model (TriDetModel): TriDet model for trajectory segmentation.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.lr = config["lr"]

        model_config = config["model"]
        preprocessor = I3D.load_from_checkpoint(
            model_config["preprocessor"]["checkpoint"]
        )

        self.model = TriDetModel(
            "SGP",
            "fpn",
            model_config["backbone"]["arch"],
            model_config["backbone"]["scale_factor"],
            model_config["input_dim"],
            model_config["max_seq_len"],
            model_config["max_buffer_len_factor"],
            model_config["backbone"]["n_sgp_win_size"],
            model_config["backbone"]["embd_kernel_size"],
            model_config["backbone"]["embd_dim"],
            model_config["backbone"]["emb_with_ln"],
            model_config["backbone"]["fpn_dim"],
            model_config["backbone"]["mlp_dim"],
            model_config["backbone"]["fpn_with_ln"],
            model_config["head"]["dim"],
            model_config["regression_range"],
            model_config["head"]["num_layers"],
            model_config["head"]["kernel_size"],
            model_config["head"]["boundary_kernel_size"],
            model_config["head"]["with_ln"],
            model_config["backbone"]["use_abs_pe"],
            model_config["num_bins"],
            model_config["iou_weight_power"],
            model_config["backbone"]["downsample_type"],
            model_config["preprocessor"]["input_noise"],
            model_config["backbone"]["k"],
            model_config["backbone"]["init_conv_vars"],
            True,
            model_config["num_classes"],
            model_config["train_cfg"],
            model_config["test_cfg"],
            model_config["use_i3d_features"],
            preprocessor,
        )

    def training_step(self, data: dict, data_idx: int) -> None:
        """
        Training step for the TriDet model.

        Args:
            data: Input data for training.
            data_idx: Index of the current data batch.

        Returns:
            float: Loss value for the current training step.
        """
        metrics = self.model(*data)

        for key, value in metrics.items():
            self.log(f"training/{key}", value)

        return metrics["final_loss"]

    def validation_step(self, data: dict, data_idx: int) -> None:
        """
        Validation step for the TriDet model.

        Args:
            data: Input data for validation.
            data_idx: Index of the current data batch.
        """
        results = self.model(*data)

        correct_labels = []
        abs_diffs = []

        for result, gt_segment, gt_label in zip(results, data[2], data[3]):
            # Determine top segment
            index = result["scores"].argmax()
            pred_segment = result["segments"][index]
            pred_label = result["labels"][index]
            abs_diffs.append(torch.abs(pred_segment - gt_segment).sum().cpu().item())
            correct_labels.append((pred_label != gt_label)[0].cpu().item())

        self.log(
            f"validation/sample_label_acc",
            np.array(correct_labels).mean(),
            batch_size=len(correct_labels),
            sync_dist=True,
        )
        self.log(
            f"validation/sample_segment_acc",
            np.array(abs_diffs).mean(),
            batch_size=len(correct_labels),
            sync_dist=True,
        )

    def segment_segment(self, data: dict) -> tuple:
        """
        Segment a trajectory using the TriDet model.

        Args:
            data: Input data for segmentation.

        Returns:
            tuple: Predicted segment and label.
        """
        result = self.model(*data)[0]

        index = result["scores"].argmax()
        pred_segment = result["segments"][index]
        pred_label = result["labels"][index]

        return pred_segment, pred_label

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training the TriDet model.

        Returns:
            torch.optim.Adam: Adam optimizer with the model parameters and learning rate.
        """
        return Adam(list(self.model.parameters()), lr=self.lr)
