# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry

from .post_processing import get_panoptic_segmentation
import matplotlib.pyplot as plt
from scipy.stats import entropy
import copy

__all__ = ["PanopticDeepLab", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch"]


INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""


@META_ARCH_REGISTRY.register()
class PanopticDeepLab(nn.Module):
    """
    Main class for panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        assert (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
            == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )
        images = ImageList.from_tensors(images, size_divisibility)

        features = self.backbone(images.tensor)

        losses = {}
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        else:
            targets = None
            weights = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, targets, weights)
        losses.update(sem_seg_losses)

        if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
            offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
            offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor
        else:
            center_targets = None
            center_weights = None

            offset_targets = None
            offset_weights = None

        center_results, offset_results, center_losses, offset_losses = self.ins_embed_head(
            features, center_targets, center_weights, offset_targets, offset_weights
        )
        losses.update(center_losses)
        losses.update(offset_losses)

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []

        processed_results = []
        for sem_seg_result, center_result, offset_result, input_per_image, image_size in zip(
            sem_seg_results, center_results, offset_results, batched_inputs, images.image_sizes
        ):

            sem_seg_positive = sem_seg_result[0]
            sem_seg_negative = sem_seg_result[1]
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(sem_seg_positive, image_size, height, width)
            c = sem_seg_postprocess(center_result, image_size, height, width)
            o = sem_seg_postprocess(offset_result, image_size, height, width)

            r_n = sem_seg_postprocess(sem_seg_negative, image_size, height, width)

            sem = r.argmax(dim=0, keepdim=True)
            if hasattr(self, "evaluate_ood"):
                evaluate_ood = self.evaluate_ood
            else:
                evaluate_ood = False

            if evaluate_ood:
                sum_val = r_n.sum(axis=0)

                # TODO

            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                sem,
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )

            if evaluate_ood:
                # Post-processing to get OOD panoptic segmentation.
                panoptic_image_ood, _ = get_panoptic_segmentation(
                    sem_out,
                    c,
                    o,
                    thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                    label_divisor=self.meta.label_divisor,
                    stuff_area=self.stuff_area,
                    void_label=-1,
                    threshold=self.threshold,
                    nms_kernel=self.nms_kernel,
                    top_k=self.top_k,
                )

            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": torch.squeeze(sem)})
            processed_results[-1]["sem_score"] = r
            processed_results[-1]["centre_score"] = c
            processed_results[-1]["offset_score"] = o
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)

            if evaluate_ood:
                processed_results[-1]["sem_seg_ood"] = torch.squeeze(sem_out)
                processed_results[-1]["anomaly_score"] = torch.squeeze(anomaly_score)
                panoptic_image_ood = panoptic_image_ood.squeeze(0)
                processed_results[-1]["panoptic_seg_ood"] = (panoptic_image_ood, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor(
                            [sem_scores * center_scores], device=panoptic_image.device
                        )
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results


@SEM_SEG_HEADS_REGISTRY.register()
class PanopticDeepLabSemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.top_k_percent_pixels = loss_top_k

        self.loss_weight = loss_weight
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])
        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1, activation=F.sigmoid)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        self.neg_head = copy.deepcopy(self.head)
        self.neg_predictor = Conv2d(head_channels, num_classes, kernel_size=1, activation=F.sigmoid)
        nn.init.normal_(self.neg_predictor.weight, 0, 0.001)
        nn.init.constant_(self.neg_predictor.bias, 0)

        self.num_classes = num_classes
        self.ignore_value = ignore_value

        self.BCE_loss = nn.BCELoss(reduce=False, reduction="mean")
        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=ignore_value, top_k_percent_pixels=loss_top_k)
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        yp, yn = self.layers(features)
        if self.training:
            return None, self.losses(yp, yn, targets, weights)
        else:
            yp = F.interpolate(
                yp, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )

            yn = F.interpolate(
                yn, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )

            return (yp, yn), {}

    def layers(self, features):
        assert self.decoder_only
        yp = super().layers(features)
        yp = self.head(yp)
        yp = self.predictor(yp)

        yn = super().layers(features)
        yn = self.neg_head(yn)
        yn = self.neg_predictor(yn)

        '''ent = entropy(F.softmax(y, dim=1).detach().cpu().numpy(), axis=1) / np.log(19)
        plt.imshow(torch.squeeze(ent).detach().cpu().numpy())
        plt.show()'''
        return yp, yn

    def losses(self, positive_predictions, negative_predictions, targets, weights=None):
        positive_predictions = F.interpolate(
            positive_predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )

        negative_predictions = F.interpolate(
            negative_predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )

        positive_targets = torch.clone(targets)
        negative_targets = torch.clone(targets)

        positive_targets[targets == self.ignore_value] = self.num_classes
        enc_positive = torch.eye(self.num_classes + 1)[positive_targets][..., :-1]
        enc_positive[targets == self.ignore_value] = torch.zeros(self.num_classes)
        enc_positive = enc_positive.permute(0, 3, 1, 2).contiguous()

        negative_targets[targets == self.ignore_value] = self.num_classes
        enc_negative = torch.eye(self.num_classes + 1)[negative_targets][..., :-1]
        enc_negative[targets == self.ignore_value] = torch.zeros(self.num_classes)
        enc_negative = enc_negative.permute(0, 3, 1, 2).contiguous()


        enc_positive_target = enc_positive.to(targets.device)
        enc_negative_target = enc_negative.to(targets.device)

        #enc_weights = enc_target * 1
        #enc_weights[enc_weights==0] = 1
        positive_weights = weights.clone()
        positive_weights = positive_weights.unsqueeze(dim=1)
        positive_weights = positive_weights.repeat(1,19,1,1)
        positive_weights[enc_positive_target==0] =  positive_weights[enc_positive_target==0] * 0.5

        negative_weights = weights.clone()
        negative_weights = negative_weights.unsqueeze(dim=1)
        negative_weights = negative_weights.repeat(1, 19, 1, 1)
        negative_weights[enc_negative_target == 1] = negative_weights[enc_negative_target == 1] * 0.5


        positive_losses = self.BCE_loss(positive_predictions, enc_positive_target) * positive_weights
        positive_losses = positive_losses.contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            positive_losses = positive_losses.mean()
        else:
            top_k_pixels = int(self.top_k_percent_pixels * positive_losses.numel())
            positive_losses, _ = torch.topk(positive_losses, top_k_pixels)
            positive_losses = positive_losses.mean()

        negative_losses = self.BCE_loss(negative_predictions, enc_negative_target) * negative_weights
        negative_losses = negative_losses.contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            negative_losses = negative_losses.mean()
        else:
            top_k_pixels = int(self.top_k_percent_pixels * negative_losses.numel())
            negative_losses, _ = torch.topk(negative_losses, top_k_pixels)
            negative_losses = negative_losses.mean()

        losses = {"loss_pos_seg": positive_losses * 5.0, "loss_neg_seg":  negative_losses * 5.0}
        return losses


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


@INS_EMBED_BRANCHES_REGISTRY.register()
class PanopticDeepLabInsEmbedHead(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset = self.layers(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            center = F.interpolate(
                center, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            offset = (
                F.interpolate(
                    offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
            )
            return center, offset, {}, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return center, offset

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses