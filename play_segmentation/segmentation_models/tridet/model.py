import math

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from play_segmentation.utils.nn import ImageEncoderCalvin

"""
Code adapted from TriDet by Dingfen Shi:
https://github.com/dingfengshi/TriDet

MIT License

Copyright (c) 2023 Dingfeng Shi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def make_generator(name, **kwargs):
    generator = generators[name](**kwargs)
    return generator


def make_backbone(name, **kwargs):
    backbone = backbones[name](**kwargs)
    return backbone


def make_neck(name, **kwargs):
    neck = necks[name](**kwargs)
    return neck


# backbone (e.g., conv / transformer)
backbones = {}


def register_backbone(name):
    def decorator(cls):
        backbones[name] = cls
        return cls

    return decorator


# neck (e.g., FPN)
necks = {}


def register_neck(name):
    def decorator(cls):
        necks[name] = cls
        return cls

    return decorator


# location generator (point, segment, etc)
generators = {}


def register_generator(name):
    def decorator(cls):
        generators[name] = cls
        return cls

    return decorator


class TriDetModel(nn.Module):
    """
    Transformer based model for single stage action localization
    """

    def __init__(
        self,
        backbone_type,  # a string defines which backbone we use
        fpn_type,  # a string defines which fpn we use
        backbone_arch,  # a tuple defines # layers in embed / stem / branch
        scale_factor,  # scale factor between branch layers
        input_dim,  # input feat dim
        max_seq_len,  # max sequence length (used for training)
        max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
        n_sgp_win_size,  # window size w for sgp
        embd_kernel_size,  # kernel size of the embedding network
        embd_dim,  # output feat channel of the embedding network
        embd_with_ln,  # attach layernorm to embedding network
        fpn_dim,  # feature dim on FPN,
        sgp_mlp_dim,  # the numnber of dim in SGP
        fpn_with_ln,  # if to apply layer norm at the end of fpn
        head_dim,  # feature dim for head
        regression_range,  # regression range on each level of FPN
        head_num_layers,  # number of layers in the head (including the classifier)
        head_kernel_size,  # kernel size for reg/cls heads
        boudary_kernel_size,  # kernel size for boundary heads
        head_with_ln,  # attache layernorm to reg/cls heads
        use_abs_pe,  # if to use abs position encoding
        num_bins,  # the bin number in Trident-head (exclude 0)
        iou_weight_power,  # the power of iou weight in loss
        downsample_type,  # how to downsample feature in FPN
        input_noise,  # add gaussian noise with the variance, play a similar role to position embedding
        k,  # the K in SGP
        init_conv_vars,  # initialization of gaussian variance for the weight in SGP
        use_trident_head,  # if use the Trident-head
        num_classes,  # number of action classes
        train_cfg,  # other cfg for training
        test_cfg,  # other cfg for testing
        use_i3d_features,  # use i3d features
        preprocessor,  # Transforms Video data into features
    ):
        super().__init__()

        # Preprocessing Features
        self.use_i3d_features = use_i3d_features
        if self.use_i3d_features:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = ImageEncoderCalvin(False)

        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1] + 1)]

        self.input_noise = input_noise

        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.iou_weight_power = iou_weight_power
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_sgp_win_size, int):
            self.sgp_win_size = [n_sgp_win_size] * len(self.fpn_strides)
        else:
            assert len(n_sgp_win_size) == len(self.fpn_strides)
            self.sgp_win_size = n_sgp_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.sgp_win_size)):
            stride = s * w if w > 1 else s
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg["center_sample"]  # ?
        assert self.train_center_sample in ["radius", "none"]  # ?
        self.train_center_sample_radius = train_cfg["center_sample_radius"]  # ?
        self.train_loss_weight = train_cfg["loss_weight"]  # ?
        self.train_cls_prior_prob = train_cfg["cls_prior_prob"]  # ?
        self.train_dropout = train_cfg["dropout"]  # ?
        self.train_droppath = train_cfg["droppath"]  # ?
        self.train_label_smoothing = train_cfg["label_smoothing"]  # ?

        # test time config
        self.test_pre_nms_thresh = test_cfg["pre_nms_thresh"]  # ?
        self.test_pre_nms_topk = test_cfg["pre_nms_topk"]  # ?
        self.test_iou_threshold = test_cfg["iou_threshold"]  # ?
        self.test_min_score = test_cfg["min_score"]  # ?
        self.test_max_seg_num = test_cfg["max_seg_num"]  # ?
        self.test_nms_method = test_cfg["nms_method"]  # ?
        assert self.test_nms_method in ["soft", "hard", "none"]  # ?
        self.test_duration_thresh = test_cfg["duration_thresh"]  # ?
        self.test_multiclass_nms = test_cfg["multiclass_nms"]  # ?
        self.test_nms_sigma = test_cfg["nms_sigma"]  # ?
        self.test_voting_thresh = test_cfg["voting_thresh"]  # ?

        self.num_bins = num_bins
        self.use_trident_head = use_trident_head

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ["SGP", "conv"]
        if backbone_type == "SGP":
            self.backbone = make_backbone(
                "SGP",
                **{
                    "n_in": input_dim,
                    "n_embd": embd_dim,
                    "sgp_mlp_dim": sgp_mlp_dim,
                    "n_embd_ks": embd_kernel_size,
                    "max_len": max_seq_len,
                    "arch": backbone_arch,
                    "scale_factor": scale_factor,
                    "with_ln": embd_with_ln,
                    "path_pdrop": self.train_droppath,
                    "downsample_type": downsample_type,
                    "sgp_win_size": self.sgp_win_size,
                    "use_abs_pe": use_abs_pe,
                    "k": k,
                    "init_conv_vars": init_conv_vars,
                },
            )
        else:
            self.backbone = make_backbone(
                "conv",
                **{
                    "n_in": input_dim,
                    "n_embd": embd_dim,
                    "n_embd_ks": embd_kernel_size,
                    "arch": backbone_arch,
                    "scale_factor": scale_factor,
                    "with_ln": embd_with_ln,
                },
            )

        # fpn network: convs
        assert fpn_type in ["fpn", "identity"]
        self.neck = make_neck(
            fpn_type,
            **{
                "in_channels": [embd_dim] * (backbone_arch[-1] + 1),
                "out_channel": fpn_dim,
                "scale_factor": scale_factor,
                "with_ln": fpn_with_ln,
            },
        )

        # location generator: points
        self.point_generator = make_generator(
            "point",
            **{
                "max_seq_len": max_seq_len * max_buffer_len_factor,
                "fpn_levels": len(self.fpn_strides),
                "scale_factor": scale_factor,
                "regression_range": self.reg_range,
                "strides": self.fpn_strides,
            },
        )

        # classfication and regerssion heads
        self.cls_head = ClsHead(
            fpn_dim,
            head_dim,
            self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg["head_empty_cls"],
        )

        if use_trident_head:
            self.start_head = ClsHead(
                fpn_dim,
                head_dim,
                self.num_classes,
                kernel_size=boudary_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=train_cfg["head_empty_cls"],
                detach_feat=True,
            )
            self.end_head = ClsHead(
                fpn_dim,
                head_dim,
                self.num_classes,
                kernel_size=boudary_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=train_cfg["head_empty_cls"],
                detach_feat=True,
            )

            self.reg_head = RegHead(
                fpn_dim,
                head_dim,
                len(self.fpn_strides),
                kernel_size=head_kernel_size,
                num_layers=head_num_layers,
                with_ln=head_with_ln,
                num_bins=num_bins,
            )
        else:
            self.reg_head = RegHead(
                fpn_dim,
                head_dim,
                len(self.fpn_strides),
                kernel_size=head_kernel_size,
                num_layers=head_num_layers,
                with_ln=head_with_ln,
                num_bins=0,
            )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg["init_loss_norm"]
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def decode_offset(self, out_offsets, pred_start_neighbours, pred_end_neighbours):
        # decode the offset value from the network output
        # If a normal regression head is used, the offsets is predicted directly in the out_offsets.
        # If the Trident-head is used, the predicted offset is calculated using the value from
        # center offset head (out_offsets), start boundary head (pred_left) and end boundary head (pred_right)

        if not self.use_trident_head:
            if self.training:
                out_offsets = torch.cat(out_offsets, dim=1)
            return out_offsets

        else:
            # Make an adaption for train and validation, when training, the out_offsets is a list with feature outputs
            # from each FPN level. Each feature with shape [batchsize, T_level, (Num_bin+1)x2].
            # For validation, the out_offsets is a feature with shape [T_level, (Num_bin+1)x2]
            if self.training:
                out_offsets = torch.cat(out_offsets, dim=1)
                out_offsets = out_offsets.view(out_offsets.shape[:2] + (2, -1))
                pred_start_neighbours = torch.cat(pred_start_neighbours, dim=1)
                pred_end_neighbours = torch.cat(pred_end_neighbours, dim=1)

                pred_left_dis = torch.softmax(
                    pred_start_neighbours + out_offsets[:, :, :1, :], dim=-1
                )
                pred_right_dis = torch.softmax(
                    pred_end_neighbours + out_offsets[:, :, 1:, :], dim=-1
                )

            else:
                out_offsets = out_offsets.view(out_offsets.shape[0], 2, -1)
                pred_left_dis = torch.softmax(
                    pred_start_neighbours + out_offsets[None, :, 0, :], dim=-1
                )
                pred_right_dis = torch.softmax(
                    pred_end_neighbours + out_offsets[None, :, 1, :], dim=-1
                )

            max_range_num = pred_left_dis.shape[-1]

            left_range_idx = torch.arange(
                max_range_num - 1,
                -1,
                -1,
                device=pred_start_neighbours.device,
                dtype=torch.float,
            ).unsqueeze(-1)
            right_range_idx = torch.arange(
                max_range_num, device=pred_end_neighbours.device, dtype=torch.float
            ).unsqueeze(-1)

            pred_left_dis = pred_left_dis.masked_fill(torch.isnan(pred_right_dis), 0)
            pred_right_dis = pred_right_dis.masked_fill(torch.isnan(pred_right_dis), 0)

            # calculate the value of expectation for the offset:
            decoded_offset_left = torch.matmul(pred_left_dis, left_range_idx)
            decoded_offset_right = torch.matmul(pred_right_dis, right_range_idx)

            return torch.cat([decoded_offset_left, decoded_offset_right], dim=-1)

    def forward(self, videos, obs_lengths, gt_segments, gt_labels):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(videos, obs_lengths)
        n_videos = batched_inputs.shape[0]

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)

        if self.use_trident_head:
            out_lb_logits = self.start_head(fpn_feats, fpn_masks)
            out_rb_logits = self.end_head(fpn_feats, fpn_masks)
        else:
            out_lb_logits = None
            out_rb_logits = None

        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels
            )

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits,
                out_offsets,
                gt_cls_labels,
                gt_offsets,
                out_lb_logits,
                out_rb_logits,
                gt_labels,
                gt_segments,
            )
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                n_videos,
                points,
                fpn_masks,
                out_cls_logits,
                out_offsets,
                out_lb_logits,
                out_rb_logits,
            )
            return results

    @torch.no_grad()
    def preprocessing(self, videos, obs_lengths):
        """
        Generate batched features and masks from a list of dict items
        videos: BxTxCxHxW
        obs_lengths: T
        """

        if self.use_i3d_features:
            batched_inputs = self.preprocessor.encode(videos)
        else:
            B = videos.shape[0]
            videos = rearrange(videos, "b t c w h -> (b t) h w c")
            batched_inputs = self.preprocessor(videos)
            batched_inputs = rearrange(batched_inputs, "(b t) d -> b d t", b=B)

        if self.input_noise > 0:
            # trick, adding noise slightly increases the variability between input features.
            noise = torch.randn_like(batched_inputs) * self.input_noise
            batched_inputs += noise

        # generate the mask
        max_length = obs_lengths.max()
        batched_masks = (
            torch.arange(max_length).to(obs_lengths.device)[None, :]
            < obs_lengths[:, None]
        )
        batched_masks = batched_masks.unsqueeze(1)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)

        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == "radius":
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = (
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            )
            t_maxs = (
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            )
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] - torch.maximum(
                t_mins, gt_segs[:, :, 0]
            )
            cb_dist_right = (
                torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
            )
            # F T x N x 2
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0

        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None]),
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
        lens.masked_fill_(inside_regress_range == 0, float("inf"))
        # print(lens)
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # print(min_len)
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float("inf"))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        # print(min_len_mask)
        # print(min_len_mask.shape)
        # print(gt_label)

        gt_label_one_hot = (
            F.one_hot(gt_label, self.num_classes).to(reg_targets.dtype).squeeze(0)
        )

        # print(gt_label_one_hot)
        # print(min_len_mask)
        # print(gt_label_one_hot.shape)
        # print(min_len_mask.shape)
        cls_targets = min_len_mask @ gt_label_one_hot
        # print(cls_targets)
        # print(cls_targets.shape)
        # print(cls_targets[min_len_mask.bool().squeeze(-1)])

        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds

        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
        self,
        fpn_masks,
        out_cls_logits,
        out_offsets,
        gt_cls_labels,
        gt_offsets,
        out_start,
        out_end,
        gt_labels,
        gt_segments,
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        if self.use_trident_head:
            out_start_logits = []
            out_end_logits = []
            for i in range(len(out_start)):
                x = (
                    F.pad(out_start[i], (self.num_bins, 0), mode="constant", value=0)
                ).unsqueeze(
                    -1
                )  # pad left
                x_size = list(x.size())  # bz, cls_num, T+num_bins, 1
                x_size[-1] = self.num_bins + 1  # bz, cls_num, T+num_bins, num_bins + 1
                x_size[-2] = (
                    x_size[-2] - self.num_bins
                )  # bz, cls_num, T+num_bins, num_bins + 1
                x_stride = list(x.stride())
                x_stride[-2] = x_stride[-1]

                x = x.as_strided(size=x_size, stride=x_stride)

                out_start_logits.append(x.permute(0, 2, 1, 3))

                x = (
                    F.pad(out_end[i], (0, self.num_bins), mode="constant", value=0)
                ).unsqueeze(
                    -1
                )  # pad right
                x = x.as_strided(size=x_size, stride=x_stride)
                out_end_logits.append(x.permute(0, 2, 1, 3))

        else:
            out_start_logits = None
            out_end_logits = None

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        # print(f"GT class labels {gt_cls_labels}")
        gt_cls = torch.stack(gt_cls_labels)

        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        # print(pos_mask)

        decoded_offsets = self.decode_offset(
            out_offsets, out_start_logits, out_end_logits
        )  # bz, stack_T, num_class, 2
        decoded_offsets = decoded_offsets[pos_mask]

        if self.use_trident_head:
            # the boundary head predicts the classification score for each categories.
            pred_offsets = decoded_offsets[gt_cls[pos_mask].bool()]
            # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
            vid = torch.where(gt_cls[pos_mask])[0]
            gt_offsets = torch.stack(gt_offsets)[pos_mask][vid]
        else:
            pred_offsets = decoded_offsets
            gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out

        gt_target = gt_cls[valid_mask]

        # optimal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        # print(f"GT target {gt_target}")
        # print(f"Out cls logits {out_cls_logits}")
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask], gt_target, reduction="none"
        )

        if self.use_trident_head:
            # couple the classification loss with iou score
            iou_rate = ctr_giou_loss_1d(pred_offsets, gt_offsets, reduction="none")
            rated_mask = gt_target > self.train_label_smoothing / (self.num_classes + 1)

            cls_loss_without_iou = cls_loss.sum()
            cls_loss[rated_mask] *= (1 - iou_rate) ** self.iou_weight_power

        B, D = valid_mask.shape

        predicted_classes = torch.cat(out_cls_logits, dim=1)[valid_mask]
        predicted_classes = predicted_classes.reshape(B, -1, self.num_classes)
        predicted_classes = predicted_classes.flatten(start_dim=1, end_dim=2)
        predicted_classes = predicted_classes.argmax(dim=-1)
        predicted_classes = (
            predicted_classes
            - (torch.div(predicted_classes, 34, rounding_mode="floor")) * 34
        )

        gt_classes = torch.tensor([elem[0].item() for elem in gt_labels]).to(
            predicted_classes.device
        )
        cls_acc = (predicted_classes == gt_classes).float().mean()

        # class_acc = predicted_classes==gt_targets
        cls_loss = cls_loss.sum()
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(pred_offsets, gt_offsets, reduction="sum")
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "final_loss": final_loss,
            "iou_rate": iou_rate.mean(),
            "cls_loss_without_iou": cls_loss_without_iou,
            "cls_accuracy": cls_acc,
        }

    @torch.no_grad()
    def inference(
        self,
        n_videos,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
        out_lb_logits,
        out_rb_logits,
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx in range(n_videos):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]

            if self.use_trident_head:
                lb_logits_per_vid = [x[idx] for x in out_lb_logits]
                rb_logits_per_vid = [x[idx] for x in out_rb_logits]
            else:
                lb_logits_per_vid = [None for x in range(len(out_cls_logits))]
                rb_logits_per_vid = [None for x in range(len(out_cls_logits))]

            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points,
                fpn_masks_per_vid,
                cls_logits_per_vid,
                offsets_per_vid,
                lb_logits_per_vid,
                rb_logits_per_vid,
            )
            results.append(results_per_vid)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
        lb_logits_per_vid,
        rb_logits_per_vid,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i, sb_cls_i, eb_cls_i in zip(
            out_cls_logits,
            out_offsets,
            points,
            fpn_masks,
            lb_logits_per_vid,
            rb_logits_per_vid,
        ):
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = pred_prob > self.test_pre_nms_thresh
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode="floor")
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. For efficiency, pad the boarder head with num_bins zeros (Pad left for start branch and Pad right
            # for end branch). Then we re-arrange the output of boundary branch to [class_num, T, num_bins + 1 (the
            # neighbour bin for each instant)]. In this way, the output can be directly added to the center offset
            # later.
            if self.use_trident_head:
                # pad the boarder
                x = (
                    F.pad(sb_cls_i, (self.num_bins, 0), mode="constant", value=0)
                ).unsqueeze(
                    -1
                )  # pad left
                x_size = list(x.size())  # cls_num, T+num_bins, 1
                x_size[-1] = self.num_bins + 1
                x_size[-2] = x_size[-2] - self.num_bins  # cls_num, T, num_bins + 1
                x_stride = list(x.stride())
                x_stride[-2] = x_stride[-1]

                pred_start_neighbours = x.as_strided(size=x_size, stride=x_stride)

                x = (
                    F.pad(eb_cls_i, (0, self.num_bins), mode="constant", value=0)
                ).unsqueeze(
                    -1
                )  # pad right
                pred_end_neighbours = x.as_strided(size=x_size, stride=x_stride)
            else:
                pred_start_neighbours = None
                pred_end_neighbours = None

            decoded_offsets = self.decode_offset(
                offsets_i, pred_start_neighbours, pred_end_neighbours
            )

            # pick topk output from the prediction
            if self.use_trident_head:
                offsets = decoded_offsets[cls_idxs, pt_idxs]
            else:
                offsets = decoded_offsets[pt_idxs]

            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]

            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            # print(seg_right,seg_left)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]

        results = {"segments": segs_all, "scores": scores_all, "labels": cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid["video_id"]
            fps = results_per_vid["fps"]
            vlen = results_per_vid["duration"]
            stride = results_per_vid["feat_stride"]
            nframes = results_per_vid["feat_num_frames"]
            # 1: unpack the results and move to CPU
            segs = results_per_vid["segments"].detach().cpu()
            scores = results_per_vid["scores"].detach().cpu()
            labels = results_per_vid["labels"].detach().cpu()
            if self.test_nms_method != "none":
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs,
                    scores,
                    labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == "soft"),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh,
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen
            # 4: repack the results
            processed_results.append(
                {"video_id": vidx, "segments": segs, "scores": scores, "labels": labels}
            )

        return processed_results


@register_backbone("SGP")
class SGPBackbone(nn.Module):
    """
    A backbone that combines SGP layer with transformers
    """

    def __init__(
        self,
        n_in,  # input feature dimension
        n_embd,  # embedding dimension (after convolution)
        sgp_mlp_dim,  # the numnber of dim in SGP
        n_embd_ks,  # conv kernel size of the embedding network
        max_len,  # max sequence length
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        scale_factor=2,  # dowsampling rate for the branch,
        with_ln=False,  # if to attach layernorm after conv
        path_pdrop=0.0,  # droput rate for drop path
        downsample_type="max",  # how to downsample feature in FPN
        sgp_win_size=[-1] * 6,  # size of local window for mha
        k=1.5,  # the K in SGP
        init_conv_vars=1,  # initialization of gaussian variance for the weight in SGP
        use_abs_pe=False,  # use absolute position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(sgp_win_size) == (1 + arch[2])
        self.arch = arch
        self.sgp_win_size = sgp_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(
                MaskedConv1D(
                    in_channels,
                    n_embd,
                    n_embd_ks,
                    stride=1,
                    padding=n_embd_ks // 2,
                    bias=(not with_ln),
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                SGPBlock(
                    n_embd,
                    1,
                    1,
                    n_hidden=sgp_mlp_dim,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                SGPBlock(
                    n_embd,
                    self.sgp_win_size[1 + idx],
                    self.scale_factor,
                    path_pdrop=path_pdrop,
                    n_hidden=sgp_mlp_dim,
                    downsample_type=downsample_type,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
            )
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(self.pos_embd, T, mode="linear", align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem network
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x,)
        out_masks += (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=T // self.stride, mode="nearest"
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class ClsHead(nn.Module):
    """
    1D Conv heads for classification
    """

    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls=[],
        detach_feat=False,
    ):
        super().__init__()
        self.act = act_layer()
        self.detach_feat = detach_feat

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln),
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size, stride=1, padding=kernel_size // 2
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            if self.detach_feat:
                cur_out = cur_feat.detach()
            else:
                cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits.squeeze(1),)

        # fpn_masks remains the same
        return out_logits


class RegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        num_bins=16,
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln),
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        self.offset_head = MaskedConv1D(
            feat_dim,
            2 * (num_bins + 1),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        # fpn_masks remains the same
        return out_offsets


class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,  # input feature channels, len(in_channels) = # levels
        out_channel,  # output feature channel
        scale_factor=2.0,  # downsampling rate between two fpn levels
        start_level=0,  # start fpn level
        end_level=-1,  # end fpn level
        with_ln=True,  # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i + self.start_level] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x,)

        return fpn_feats, fpn_masks


@register_generator("point")
class PointGenerator(nn.Module):
    """
    A generator for temporal "points"

    max_seq_len can be much larger than the actual seq length
    """

    def __init__(
        self,
        max_seq_len,  # max sequence length that the generator will buffer
        fpn_levels,  # number of fpn levels
        scale_factor,  # scale factor between two fpn levels
        regression_range,  # regression range (on feature grids)
        strides,  # stride of fpn levels
        use_offset=False,  # if to align the points at grid centers
    ):
        super().__init__()
        # sanity check, # fpn levels and length divisible
        assert len(regression_range) == fpn_levels
        assert max_seq_len % scale_factor ** (fpn_levels - 1) == 0

        # save params
        self.max_seq_len = max_seq_len
        self.fpn_levels = fpn_levels
        self.scale_factor = scale_factor
        self.regression_range = regression_range
        self.strides = strides
        self.use_offset = use_offset

        # generate all points and buffer the list
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # loop over all points at each pyramid level

        for l in range(self.fpn_levels):
            stride = self.strides[l]
            reg_range = torch.as_tensor(self.regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points = torch.arange(0, self.max_seq_len, stride)[:, None]
            # add offset if necessary (not in our current model)
            if self.use_offset:
                points += 0.5 * stride
            # pad the time stamp with additional regression range / stride
            reg_range = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            # size: T x 4 (ts, reg_range, stride)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))

        return BufferList(points_list)

    def forward(self, feats):
        # feats will be a list of torch tensors
        assert len(feats) == self.fpn_levels
        pts_list = []
        feat_lens = [feat.shape[-1] for feat in feats]
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            assert (
                feat_len <= buffer_pts.shape[0]
            ), "Reached max buffer length for point generator"
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list


class SGPBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=3,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        k=1.5,  # k
        group=1,  # group for cnn
        n_out=None,  # output dimension, if None, set to input dim
        n_hidden=None,  # hidden dim for mlp
        path_pdrop=0.0,  # drop path rate
        act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
        downsample_type="max",
        init_conv_vars=1,  # init gaussian variance for the weight
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(
            n_embd,
            n_embd,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=n_embd,
        )
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(
            n_embd,
            n_embd,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=n_embd,
        )
        self.convkw = nn.Conv1d(
            n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd
        )
        self.global_fc = nn.Conv1d(
            n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd
        )

        # input
        if n_ds_stride > 1:
            if downsample_type == "max":
                kernel_size, stride, padding = (
                    n_ds_stride + 1,
                    n_ds_stride,
                    (n_ds_stride + 1) // 2,
                )
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding
                )
                self.stride = stride
            elif downsample_type == "avg":
                self.downsample = nn.Sequential(
                    nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                    nn.Conv1d(n_embd, n_embd, 1, 1, 0),
                )
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)

        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode="trunc") + T % self.stride,
            mode="nearest",
        ).detach()

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (convw + convkw) * psi + out

        out = x * out_mask + self.drop_path_out(out)

        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ):
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

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32), requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)), requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers

    Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


@torch.jit.script
def ctr_giou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


@register_neck("fpn")
class FPN1D(nn.Module):
    """
    Feature pyramid network
    """

    def __init__(
        self,
        in_channels,  # input feature channels, len(in_channels) = # levels
        out_channel,  # output feature channel
        scale_factor=2.0,  # downsampling rate between two fpn levels
        start_level=0,  # start fpn level
        end_level=-1,  # end fpn level
        with_ln=True,  # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(in_channels[i], out_channel, 1, bias=(not with_ln))
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=(not with_ln),
                groups=out_channel,
            )
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=self.scale_factor, mode="nearest"
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        for i in range(used_backbone_levels):
            x, _ = self.fpn_convs[i](laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x,)

        return fpn_feats, fpn_masks
