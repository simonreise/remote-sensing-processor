"""Semantic segmentation models."""

from pydantic import InstanceOf
from typing import Any, Optional, Union

import json
import tempfile
import warnings
from pathlib import Path

import kornia

import segmentation_models_pytorch
import torch
import torchgeo.models
import torchvision
import transformers

import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from remote_sensing_processor.common.types import SKLModel, TorchNNModel


def load_backbone(bb: str, input_shape: int, input_dims: int) -> transformers.PretrainedConfig:
    """Load backbone for a HF Transformers model."""
    if bb == "BEiT":
        backbone = transformers.BeitConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "BiT":
        backbone = transformers.BitConfig(
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "ConvNeXT":
        backbone = transformers.ConvNextConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "ConvNeXTV2":
        backbone = transformers.ConvNextV2Config(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    # Currently not supported because there's no natten package in conda and no windows support
    elif bb == "DiNAT":
        backbone = transformers.DinatConfig(
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "DINOV2":
        backbone = transformers.Dinov2Config(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "DINOV2WithRegisters":
        backbone = transformers.Dinov2WithRegistersConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "DINOV3ViT":
        backbone = transformers.DINOv3ViTConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "DINOV3ConvNeXT":
        backbone = transformers.DINOv3ConvNextConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "FocalNet":
        backbone = transformers.FocalNetConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "HGNet-V2":
        backbone = transformers.HGNetV2Config(
            num_channels=input_dims,
            stem_channels=[input_dims, 32, 48],
        )
        backbone.out_features = ["stage2", "stage3", "stage4"]
    elif bb == "Hiera":
        backbone = transformers.HieraConfig(
            image_size=[input_shape, input_shape],
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "LW-DETR":
        backbone = transformers.LwDetrViTConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "MaskFormer-Swin":
        backbone = transformers.MaskFormerSwinConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "Pixio":
        backbone = transformers.PixioConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "PVTV2":
        backbone = transformers.PvtV2Config(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "ResNet":
        backbone = transformers.ResNetConfig(
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "RT-DETR-ResNet":
        backbone = transformers.RTDetrResNetConfig(
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "Swin":
        backbone = transformers.SwinConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "SwinV2":
        backbone = transformers.Swinv2Config(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    elif bb == "ViTDet":
        backbone = transformers.VitDetConfig(
            image_size=input_shape,
            num_channels=input_dims,
        )
        backbone.out_features = ["stage1", "stage2", "stage3", "stage4"]
    else:
        warnings.warn(
            bb + " is not one of the default backbones. Trying to load timm backbone with the requested name.",
            stacklevel=2,
        )
        backbone = transformers.TimmBackboneConfig(
            backbone=bb,
            num_channels=input_dims,
        )
    return backbone


def get_farseg_weights(bb: str, weights: str) -> Optional[InstanceOf[torchvision.models._api.WeightsEnum]]:
    """Load Farseg backbone weights."""
    if weights is not None:
        if bb is None or bb == "resnet50":
            if hasattr(torchvision.models.ResNet50_Weights, weights):
                weights = getattr(torchvision.models.ResNet50_Weights, weights)
            else:
                weights = None
        elif bb == "resnet18":
            if hasattr(torchvision.models.ResNet18_Weights, weights):
                weights = getattr(torchvision.models.ResNet18_Weights, weights)
            else:
                weights = None
        elif bb == "resnet34":
            if hasattr(torchvision.models.ResNet34_Weights, weights):
                weights = getattr(torchvision.models.ResNet34_Weights, weights)
            else:
                weights = None
        elif bb == "resnet101":
            if hasattr(torchvision.models.ResNet101_Weights, weights):
                weights = getattr(torchvision.models.ResNet101_Weights, weights)
            else:
                weights = None
        else:
            weights = None
    return weights


class TransformersModel(torch.nn.Module):
    """A custom class that includes data pre- and post-processing for Transformers models."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        processor: Union[transformers.BaseImageProcessor, transformers.OneFormerProcessor],
        input_shape: int,
        y_nodata: Optional[int] = None,
        detr: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.detr = detr
        self.input_shape = input_shape
        self.y_nodata = y_nodata

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model."""
        x = batch["x"]
        y = batch.get("y")

        # Setting up x
        inputs = {"images": x, "input_data_format": "channels_first", "return_tensors": "pt"}
        # Setting up y if processor can process it
        if y is not None and not self.detr:
            inputs["segmentation_maps"] = y
        # Oneformer also requires tokenized tasks as inputs, task is semantic
        if isinstance(self.model, transformers.OneFormerForUniversalSegmentation):
            inputs["task_inputs"] = ["semantic"] * x.shape[0]
        # Process
        inputs = self.processor(**inputs)
        # Move Oneformer task inputs to the correct device
        # TODO: Remove if something made here https://github.com/huggingface/transformers/issues/42722
        if isinstance(self.model, transformers.OneFormerForUniversalSegmentation):
            inputs["task_inputs"] = inputs["task_inputs"].to(inputs["pixel_values"].device)
            inputs["text_inputs"] = inputs["text_inputs"].to(inputs["pixel_values"].device)
        # Add DETR annotations
        if y is not None and self.detr:
            inputs["labels"] = [
                self.prepare_detr_annotation(
                    img,
                    self.y_nodata,
                    self.input_shape,
                )
                for img in y
            ]
        # Predict
        pred = self.model(**inputs)
        # Get loss
        loss = pred.loss
        # Postprocess
        pred = self.processor.post_process_semantic_segmentation(
            pred,
            target_sizes=[(self.input_shape, self.input_shape)] * x.shape[0],
            return_segmentation_scores=True,
        )
        pred = [x["segmentation_scores"] for x in pred]
        pred = torch.stack(pred)
        return pred, loss

    def prepare_detr_annotation(self, sem_seg: torch.Tensor, y_nodata: Optional[int], input_shape: int) -> dict:
        """Function that converts semantic segmentation maps to DETR annotations."""
        from transformers.models.detr.image_processing_detr import DetrImageProcessor, masks_to_boxes

        annotation = {}

        # Converting semantic segmentation map to panoptic
        panoptic_seg = torch.zeros_like(sem_seg, dtype=torch.int32)  # Output array
        unique_id = 1
        labels = []

        for class_label in torch.unique(sem_seg):
            class_label = int(class_label.item())
            if class_label != y_nodata:
                mask = sem_seg == class_label  # Get all pixels for this class

                # Use Kornia connected components function for connected components
                labeled_array = kornia.contrib.connected_components(mask[None, ...].float(), num_iterations=150)[0]

                labeled_array = torch.where(mask, labeled_array + 1, 0)  # Making sure non-class-label areas are 0
                num_features = len(torch.unique(labeled_array))  # Getting number of unique features
                for i in range(1, num_features):  # Ignore background label 0
                    panoptic_seg[labeled_array == torch.unique(labeled_array)[i].item()] = unique_id
                    unique_id += 1
                    labels.append(class_label)

        ids = torch.unique(panoptic_seg)
        ids = ids[ids != 0]
        panoptic_seg = panoptic_seg == ids[:, None, None]
        panoptic_seg = panoptic_seg.to(torch.bool)

        annotation["masks"] = panoptic_seg
        annotation["class_labels"] = torch.tensor(labels, device=sem_seg.device).long()
        annotation["boxes"] = masks_to_boxes(panoptic_seg)
        # noinspection PyTypeChecker
        return DetrImageProcessor.normalize_annotation(None, annotation, (input_shape, input_shape))


class TorchVisionModel(torch.nn.Module):
    """A custom class that includes data pre- and post-processing for TorchVision models."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model."""
        pred = self.model(batch["x"])
        if isinstance(pred, dict):
            pred = pred["out"]
        return pred, None


class SMPModel(torch.nn.Module):
    """A custom class that includes data pre- and post-processing for SMP models."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model."""
        pred = self.model(batch["x"])
        return pred, None


class TorchGeoModel(torch.nn.Module):
    """A custom class that includes data pre- and post-processing for TorchGeo models."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model."""
        pred = self.model(batch["x"])
        return pred, None


class CustomModel(torch.nn.Module):
    """A custom class that includes data pre- and post-processing for custom models."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model."""
        pred = self.model(batch["x"])
        return pred, None


class SemanticSegmentationModels:
    """Semantic segmentation models basic class."""

    model: Union[TorchNNModel, SKLModel]
    model_name: str
    input_shape: int
    input_dims: int
    num_classes: int
    y_nodata: Optional[int]

    def load_model(
        self,
        model_name: str,
        bb: Optional[str],
        weights: Optional[str],
        **kwargs: Any,
    ) -> TorchNNModel:
        """Load a Torch-based semantic segmentation model."""
        id2label = {i: f"label_{i}" for i in range(self.num_classes)}
        label2id = {v: k for k, v in id2label.items()}

        if model_name == "BEiT":
            if weights is not None:
                processor = transformers.BeitImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                model = transformers.BeitForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.BeitImageProcessor(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                config = transformers.BeitConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                config.out_indices = [3, 5, 7, 11]
                model = transformers.BeitForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "ConditionalDETR":
            if weights is not None:
                processor = transformers.ConditionalDetrImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                model = transformers.ConditionalDetrForSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.ConditionalDetrImageProcessor(
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                if bb is None:
                    config = transformers.ConditionalDetrConfig(
                        num_channels=self.input_dims,
                        id2label=id2label,
                        label2id=label2id,
                        **kwargs,
                    )
                else:
                    backbone = load_backbone(bb, input_shape=self.input_shape, input_dims=self.input_dims)
                    config = transformers.ConditionalDetrConfig(
                        backbone_config=backbone,
                        num_channels=self.input_dims,
                        id2label=id2label,
                        label2id=label2id,
                        **kwargs,
                    )
                model = transformers.ConditionalDetrForSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata, True)
        elif model_name == "Data2Vec":
            if weights is not None:
                processor = transformers.AutoImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                model = transformers.Data2VecVisionForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.BeitImageProcessor(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                config = transformers.Data2VecVisionConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    out_indices=[3, 5, 7, 11],
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.Data2VecVisionForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "DETR":
            # Processor not working with segmentation maps
            if weights is not None:
                processor = transformers.DetrImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                model = transformers.DetrForSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.DetrImageProcessor(
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                if bb is None:
                    config = transformers.DetrConfig(
                        num_channels=self.input_dims,
                        id2label=id2label,
                        label2id=label2id,
                        **kwargs,
                    )
                else:
                    backbone = load_backbone(bb, input_shape=self.input_shape, input_dims=self.input_dims)
                    config = transformers.DetrConfig(
                        backbone_config=backbone,
                        num_channels=self.input_dims,
                        id2label=id2label,
                        label2id=label2id,
                        **kwargs,
                    )
                model = transformers.DetrForSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata, True)
        elif model_name == "DPT":
            if weights is not None:
                processor = transformers.DPTImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                model = transformers.DPTForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.DPTImageProcessor(
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                config = transformers.DPTConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.DPTForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "EoMT":
            if weights is not None:
                processor = transformers.EomtImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    do_pad=False,
                    do_split_image=False,
                    ignore_index=self.y_nodata,
                )
                model = transformers.EomtForUniversalSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.EomtImageProcessor(
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    do_pad=False,
                    do_split_image=False,
                    ignore_index=self.y_nodata,
                )
                config = transformers.EomtConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    **kwargs,
                )
                model = transformers.EomtForUniversalSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "EoMT-DINOv3":
            if weights is not None:
                processor = transformers.EomtImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    do_pad=False,
                    do_split_image=False,
                    ignore_index=self.y_nodata,
                )
                model = transformers.EomtDinov3ForUniversalSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.EomtImageProcessor(
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    do_pad=False,
                    do_split_image=False,
                    ignore_index=self.y_nodata,
                )
                config = transformers.EomtDinov3Config(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    **kwargs,
                )
                model = transformers.EomtDinov3ForUniversalSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "Mask2Former":
            if weights is not None:
                processor = transformers.Mask2FormerImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_index=self.y_nodata,
                )
                config = transformers.AutoConfig.from_pretrained(
                    weights,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_value=self.y_nodata,
                    **kwargs,
                )
                if hasattr(config.backbone_config, "image_size"):
                    config.backbone_config.image_size = self.input_shape
                if hasattr(config.backbone_config, "num_channels"):
                    config.backbone_config.num_channels = self.input_dims
                model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    config=config,
                )
                model.train()
            else:
                processor = transformers.Mask2FormerImageProcessor(
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_index=self.y_nodata,
                )
                if bb is None:
                    bb = "Swin"
                backbone = load_backbone(bb, self.input_shape, self.input_dims)
                config = transformers.Mask2FormerConfig(
                    backbone_config=backbone,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_value=self.y_nodata,
                    **kwargs,
                )
                model = transformers.Mask2FormerForUniversalSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "MaskFormer":
            if weights is not None:
                processor = transformers.MaskFormerImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_index=self.y_nodata,
                )
                config = transformers.AutoConfig.from_pretrained(
                    weights,
                    id2label=id2label,
                    label2id=label2id,
                    **kwargs,
                )
                if hasattr(config.backbone_config, "image_size"):
                    config.backbone_config.image_size = self.input_shape
                if hasattr(config.backbone_config, "num_channels"):
                    config.backbone_config.num_channels = self.input_dims
                model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    config=config,
                )
                model.train()
            else:
                processor = transformers.MaskFormerImageProcessor(
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_index=self.y_nodata,
                )
                if bb is None:
                    bb = "Swin"
                backbone = load_backbone(bb, self.input_shape, self.input_dims)
                config = transformers.MaskFormerConfig(
                    backbone_config=backbone,
                    id2label=id2label,
                    label2id=label2id,
                    **kwargs,
                )
                model = transformers.MaskFormerForInstanceSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "MobileNetV2":
            if weights is not None:
                processor = transformers.MobileNetV2ImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                model = transformers.MobileNetV2ForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.MobileNetV2ImageProcessor(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                config = transformers.MobileNetV2Config(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.MobileNetV2ForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "MobileViT":
            if weights is not None:
                processor = transformers.MobileViTImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_flip_channel_order=False,
                )
                model = transformers.MobileViTForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.MobileViTImageProcessor(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_flip_channel_order=False,
                )
                config = transformers.MobileViTConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.MobileViTForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "MobileViTV2":
            if weights is not None:
                processor = transformers.MobileViTImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_flip_channel_order=False,
                )
                model = transformers.MobileViTV2ForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.MobileViTImageProcessor(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_flip_channel_order=False,
                )
                config = transformers.MobileViTV2Config(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.MobileViTV2ForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "OneFormer":
            # Creating a temporary classes JSON in a cityscapes panoptic format
            jdict = {}
            for i in range(self.num_classes):
                if i != self.y_nodata:
                    jdict[str(i)] = {"isthing": 0, "name": str(i)}
            temp = tempfile.NamedTemporaryFile(mode="w+", delete=False)  # noqa: SIM115
            json.dump(jdict, temp)
            temp.flush()
            temp.close()
            if weights is not None:
                processor = transformers.OneFormerProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_index=self.y_nodata,
                    repo_path=Path(temp.name).parent.as_posix(),
                    class_info_file=Path(temp.name).name,
                    num_text=150,
                )
                config = transformers.AutoConfig.from_pretrained(
                    weights,
                    is_training=True,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_value=self.y_nodata,
                    **kwargs,
                )
                # Setting up num_text to prevent size mismatch
                # noinspection PyUnresolvedReferences
                processor.image_processor.num_text = config.num_queries - config.text_encoder_n_ctx
                if hasattr(config.backbone_config, "image_size"):
                    config.backbone_config.image_size = self.input_shape
                if hasattr(config.backbone_config, "num_channels"):
                    config.backbone_config.num_channels = self.input_dims
                model = transformers.OneFormerForUniversalSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    config=config,
                )
                model.train()
            else:
                processor = transformers.OneFormerImageProcessor(
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_index=self.y_nodata,
                    repo_path=Path(temp.name).parent.as_posix(),
                    class_info_file=Path(temp.name).name,
                    num_text=134,
                )
                processor = transformers.OneFormerProcessor(
                    image_processor=processor,
                    tokenizer=transformers.AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
                )
                if bb is None:
                    bb = "Swin"
                backbone = load_backbone(bb, self.input_shape, self.input_dims)
                config = transformers.OneFormerConfig(
                    backbone_config=backbone,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_value=self.y_nodata,
                    is_training=True,
                    num_queries=150,
                    text_encoder_n_ctx=16,
                    **kwargs,
                )
                model = transformers.OneFormerForUniversalSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
            Path(temp.name).unlink()
        elif model_name == "SegFormer":
            if weights is not None:
                processor = transformers.SegformerImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                model = transformers.SegformerForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.SegformerImageProcessor(
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                config = transformers.SegformerConfig(
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.SegformerForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "TIPSv2DPT":
            if weights is not None:
                processor = transformers.Tipsv2DptImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                model = transformers.Tipsv2DptForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model.train()
            else:
                processor = transformers.Tipsv2DptImageProcessor(
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                config = transformers.Tipsv2DptConfig(
                    num_channels=self.input_dims,
                    id2label=id2label,
                    label2id=label2id,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.Tipsv2DptForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "UperNet":
            if weights is not None:
                processor = transformers.SegformerImageProcessor.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                config = transformers.AutoConfig.from_pretrained(
                    weights,
                    id2label=id2label,
                    label2id=label2id,
                    loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                if hasattr(config.backbone_config, "image_size"):
                    config.backbone_config.image_size = self.input_shape
                if hasattr(config.backbone_config, "num_channels"):
                    config.backbone_config.num_channels = self.input_dims
                model = transformers.UperNetForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    config=config,
                )
                model.train()
            else:
                processor = transformers.SegformerImageProcessor(
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                if bb is None:
                    bb = "Swin"
                backbone = load_backbone(bb, self.input_shape, self.input_dims)
                config = transformers.UperNetConfig(
                    backbone_config=backbone,
                    id2label=id2label,
                    label2id=label2id,
                    loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.UperNetForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "DeepLabV3":
            if bb == "MobileNet_V3_Large" or bb is None:
                if weights is not None:
                    weights = (
                        torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
                    )
                    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights, **kwargs)
                    model.classifier[4] = torch.nn.Conv2d(256, self.num_classes, kernel_size=1, stride=(1, 1))
                    model.aux_classifier[4] = torch.nn.Conv2d(10, self.num_classes, kernel_size=1, stride=(1, 1))
                else:
                    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                        num_classes=self.num_classes,
                        **kwargs,
                    )
                model.backbone["0"][0] = torch.nn.Conv2d(
                    self.input_dims,
                    16,
                    kernel_size=3,
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
            elif bb == "ResNet50":
                if weights is not None:
                    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights, **kwargs)
                    model.classifier[4] = torch.nn.Conv2d(256, self.num_classes, kernel_size=1, stride=(1, 1))
                else:
                    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=self.num_classes, **kwargs)
                model.backbone.conv1 = torch.nn.Conv2d(
                    self.input_dims,
                    64,
                    kernel_size=7,
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            elif bb == "ResNet101":
                if weights is not None:
                    weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
                    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights, **kwargs)
                    model.classifier[4] = torch.nn.Conv2d(256, self.num_classes, kernel_size=1, stride=(1, 1))
                else:
                    model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=self.num_classes, **kwargs)
                model.backbone.conv1 = torch.nn.Conv2d(
                    self.input_dims,
                    64,
                    kernel_size=7,
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            else:
                raise ValueError("Unknown backbone " + bb)
            model = TorchVisionModel(model)
        elif model_name == "FCN":
            if bb == "ResNet50" or bb is None:
                if weights is not None:
                    weights = torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                    model = torchvision.models.segmentation.fcn_resnet50(weights=weights, **kwargs)
                    model.classifier[4] = torch.nn.Conv2d(512, self.num_classes, kernel_size=1, stride=(1, 1))
                else:
                    model = torchvision.models.segmentation.fcn_resnet50(num_classes=self.num_classes, **kwargs)
                model.backbone.conv1 = torch.nn.Conv2d(
                    self.input_dims,
                    64,
                    kernel_size=7,
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            elif bb == "ResNet101":
                if weights is not None:
                    weights = torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
                    model = torchvision.models.segmentation.fcn_resnet101(weights=weights, **kwargs)
                    model.classifier[4] = torch.nn.Conv2d(512, self.num_classes, kernel_size=1, stride=(1, 1))
                else:
                    model = torchvision.models.segmentation.fcn_resnet101(num_classes=self.num_classes, **kwargs)
                model.backbone.conv1 = torch.nn.Conv2d(
                    self.input_dims,
                    64,
                    kernel_size=7,
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            else:
                raise ValueError("Unknown backbone " + bb)
            model = TorchVisionModel(model)
        elif model_name == "LRASPP":
            if weights is not None:
                weights = torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
                model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=weights, **kwargs)
                model.classifier.low_classifier = torch.nn.Conv2d(40, self.num_classes, kernel_size=1, stride=(1, 1))
                model.classifier.high_classifier = torch.nn.Conv2d(128, self.num_classes, kernel_size=1, stride=(1, 1))
            else:
                model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
                    num_classes=self.num_classes,
                    **kwargs,
                )
            model.backbone["0"][0] = torch.nn.Conv2d(
                self.input_dims,
                16,
                kernel_size=3,
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )
            model = TorchVisionModel(model)
        elif model_name == "UNet":
            model = segmentation_models_pytorch.Unet(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "UNet++":
            model = segmentation_models_pytorch.UnetPlusPlus(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "FPN":
            model = segmentation_models_pytorch.FPN(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "PSPNet":
            model = segmentation_models_pytorch.PSPNet(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "DeepLabV3_smp":
            model = segmentation_models_pytorch.DeepLabV3(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "DeepLabV3+":
            model = segmentation_models_pytorch.DeepLabV3Plus(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "Linknet":
            model = segmentation_models_pytorch.Linknet(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "MAnet":
            model = segmentation_models_pytorch.MAnet(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "PAN":
            model = segmentation_models_pytorch.PAN(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "UperNet_smp":
            model = segmentation_models_pytorch.UPerNet(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "SegFormer_smp":
            model = segmentation_models_pytorch.Segformer(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "DPT_smp":
            model = segmentation_models_pytorch.DPT(
                encoder_name=bb if bb is not None else "resnet34",
                encoder_weights=weights,
                in_channels=self.input_dims,
                classes=self.num_classes,
                **kwargs,
            )
            model = SMPModel(model)
        elif model_name == "FarSeg":
            weights = get_farseg_weights(bb, weights)
            model = torchgeo.models.FarSeg(
                backbone=bb if bb is not None else "resnet50",
                classes=self.num_classes,
                backbone_weights=weights,
            )
            model.backbone.conv1 = torch.nn.Conv2d(
                self.input_dims,
                64,
                kernel_size=7,
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            model = TorchGeoModel(model)
        else:
            raise ValueError("Unknown model " + model_name)
        return model

    def validate_model(self, model: TorchNNModel) -> TorchNNModel:
        """Check if model shapes are valid."""
        if next(iter(model.parameters())).size()[1] != self.input_dims:
            raise ValueError("model have invalid input shape")
        if list(model.parameters())[-1].size()[0] != self.num_classes:
            raise ValueError("model have invalid output shape")
        return CustomModel(model)

    def load_sklearn_model(
        self,
        model_name: str,
        bb: Optional[str],
        max_depth: Optional[int] = 6,
        **kwargs: Any,
    ) -> SKLModel:
        """Load a Sklearn-based semantic segmentation model."""
        if model_name == "Logistic Regression":
            if bb == "lbfgs" or bb is None:
                model = LogisticRegression(solver="lbfgs", n_jobs=-1, warm_start=True, verbose=1, **kwargs)
            elif bb == "liblinear":
                model = LogisticRegression(solver="liblinear", n_jobs=-1, warm_start=True, verbose=1, **kwargs)
            elif bb == "newton-cg":
                model = LogisticRegression(solver="newton-cg", n_jobs=-1, warm_start=True, verbose=1, **kwargs)
            elif bb == "newton-cholesky":
                model = LogisticRegression(solver="newton-cholesky", n_jobs=-1, warm_start=True, verbose=1, **kwargs)
            elif bb == "sag":
                model = LogisticRegression(solver="sag", n_jobs=-1, warm_start=True, verbose=1, **kwargs)
            elif bb == "saga":
                model = LogisticRegression(solver="saga", n_jobs=-1, warm_start=True, verbose=1, **kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "Ridge":
            model = RidgeClassifier(**kwargs)
        elif model_name == "SGD":
            if bb == "hinge" or bb is None:
                model = SGDClassifier(loss="hinge", warm_start=True, verbose=1000, n_jobs=-1, **kwargs)
            elif bb == "log_loss":
                model = SGDClassifier(loss="log_loss", warm_start=True, verbose=1000, n_jobs=-1, **kwargs)
            elif bb == "modified_huber":
                model = SGDClassifier(loss="modified_huber", warm_start=True, verbose=1000, n_jobs=-1, **kwargs)
            elif bb == "squared_hinge":
                model = SGDClassifier(loss="squared_hinge", warm_start=True, verbose=1000, n_jobs=-1, **kwargs)
            elif bb == "perceptron":
                model = SGDClassifier(loss="perceptron", warm_start=True, verbose=1000, n_jobs=-1, **kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "Nearest Neighbors":
            model = KNeighborsClassifier(n_jobs=-1, **kwargs)
        elif model_name == "Radius Neighbors":
            model = RadiusNeighborsClassifier(n_jobs=-1, **kwargs)
        elif model_name == "SVM":
            if bb == "rbf" or bb is None:
                model = SVC(kernel="rbf", verbose=True, **kwargs)
            elif bb == "linear":
                model = LinearSVC(verbose=1, **kwargs)
            elif bb == "poly":
                model = SVC(kernel="poly", verbose=True, **kwargs)
            elif bb == "sigmoid":
                model = SVC(kernel="sigmoid", verbose=True, **kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "Gaussian Process":
            model = GaussianProcessClassifier(n_jobs=-1, warm_start=True, **kwargs)
        elif model_name == "Naive Bayes":
            if bb == "gaussian" or bb == "Gaussian" or bb is None:
                model = GaussianNB(**kwargs)
            elif bb == "bernoulli" or bb == "Bernoulli":
                model = BernoulliNB(**kwargs)
            elif bb == "categorical" or bb == "Categorical":
                model = CategoricalNB(**kwargs)
            elif bb == "complement" or bb == "Complement":
                model = ComplementNB(**kwargs)
            elif bb == "multinomial" or bb == "Multinomial":
                model = MultinomialNB(**kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "QDA":
            model = QuadraticDiscriminantAnalysis(**kwargs)
        elif model_name == "LDA":
            model = LinearDiscriminantAnalysis(**kwargs)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(**kwargs)
        elif model_name == "Extra Tree":
            model = ExtraTreeClassifier(**kwargs)
        elif model_name == "Random Forest":
            # max_depth is by default set to 6, because it is unlimited by default and the training could be very slow.
            # To train with unlimited tree depth set max_depth = None
            model = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, warm_start=True, verbose=1, **kwargs)
        elif model_name == "Extra Trees":
            model = ExtraTreesClassifier(max_depth=max_depth, n_jobs=-1, warm_start=True, verbose=1, **kwargs)
        elif model_name == "AdaBoost":
            model = AdaBoostClassifier(**kwargs)
        elif model_name == "Gradient Boosting":
            model = HistGradientBoostingClassifier(warm_start=True, verbose=1000, validation_fraction=None, **kwargs)
        elif model_name == "Multilayer Perceptron":
            if bb == "adam" or bb is None:
                model = MLPClassifier(
                    solver="adam",
                    warm_start=True,
                    verbose=True,
                    **kwargs,
                )
            elif bb == "sgd":
                model = MLPClassifier(
                    solver="sgd",
                    warm_start=True,
                    verbose=True,
                    **kwargs,
                )
            elif bb == "lbfgs":
                model = MLPClassifier(
                    solver="lbfgs",
                    warm_start=True,
                    verbose=True,
                    **kwargs,
                )
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(tree_method="hist", verbosity=3, n_jobs=-1, **kwargs)
        elif model_name == "XGB Random Forest":
            model = xgb.XGBRFClassifier(tree_method="hist", verbosity=3, n_jobs=-1, **kwargs)
        else:
            raise ValueError("Unknown model " + model_name)
        return model

    def set_warm_start(self, **kwargs: Any) -> None:
        """Set the warm start parameters for a sklearn model."""
        if hasattr(self.model, "warm_start"):
            self.model.set_params(**{"warm_start": True})
            if self.model_name in ["Random Forest", "Extra Trees"]:
                self.model.set_params(**{"n_estimators": self.model.n_estimators * 2})
            elif self.model_name in ["Gradient Boosting"]:  # noqa SIM114
                self.model.set_params(**{"max_iter": self.model.max_iter * 2})
            elif self.model_name in ["Multilayer Perceptron"]:
                self.model.set_params(**{"max_iter": self.model.max_iter * 2})
            self.model.set_params(**kwargs)
        else:
            warnings.warn(
                self.model_name + " does not support warm_start. It will be trained from scratch.",
                stacklevel=1,
            )


pytorch_models = [
    "BEiT",
    "ConditionalDETR",
    "Data2Vec",
    "DETR",
    "DPT",
    "EoMT",
    "EoMT-DINOv3",
    "Mask2Former",
    "MaskFormer",
    "MobileNetV2",
    "MobileViT",
    "MobileViTV2",
    "OneFormer",
    "SegFormer",
    "TIPSv2DPT",
    "UperNet",
    "DeepLabV3",
    "FCN",
    "LRASPP",
    "UNet",
    "UNet++",
    "FPN",
    "PSPNet",
    "DeepLabV3_smp",
    "DeepLabV3+",
    "Linknet",
    "MAnet",
    "PAN",
    "UperNet_smp",
    "SegFormer_smp",
    "DPT_smp",
    "FarSeg",
    "Custom_Torch",
]

sklearn_models = [
    "Logistic Regression",
    "Ridge",
    "SGD",
    "Nearest Neighbors",
    "Radius Neighbors",
    "SVM",
    "Gaussian Process",
    "Naive Bayes",
    "QDA",
    "LDA",
    "Decision Tree",
    "Extra Tree",
    "Random Forest",
    "Extra Trees",
    "AdaBoost",
    "Gradient Boosting",
    "Multilayer Perceptron",
    "XGBoost",
    "XGB Random Forest",
    "Custom_Sklearn",
]
