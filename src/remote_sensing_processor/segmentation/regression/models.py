"""Regression models."""

from typing import Any, Optional, Union

import warnings

import segmentation_models_pytorch
import torch
import torchgeo.models
import torchvision
import transformers

import xgboost as xgb
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    GammaRegressor,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LassoLarsIC,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskLasso,
    OrthogonalMatchingPursuit,
    PoissonRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
    TheilSenRegressor,
    TweedieRegressor,
)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from remote_sensing_processor.common.types import SKLModel, TorchNNModel


def load_backbone(bb: str, input_shape: int, input_dims: int) -> transformers.PretrainedConfig:
    """Load backbone for a HF Transformers model."""
    if bb == "BEiT":
        backbone = transformers.BeitConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "BiT":
        backbone = transformers.BitConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "ConvNeXT":
        backbone = transformers.ConvNextConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "ConvNeXTV2":
        backbone = transformers.ConvNextV2Config(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    # Currently not supported because there's no natten package in conda and no windows support
    elif bb == "DiNAT":
        backbone = transformers.DinatConfig(
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "DINOV2":
        backbone = transformers.Dinov2Config(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "DINOV2WithRegisters":
        backbone = transformers.Dinov2WithRegistersConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "DINOV3ViT":
        backbone = transformers.DINOv3ViTConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "DINOV3ConvNeXT":
        backbone = transformers.DINOv3ConvNextConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "FocalNet":
        backbone = transformers.FocalNetConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "HGNet-V2":
        backbone = transformers.HGNetV2Config(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
            stem_channels=[input_dims, 32, 48],
        )
    elif bb == "Hiera":
        backbone = transformers.HieraConfig(
            image_size=[input_shape, input_shape],
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "MaskFormer-Swin":
        backbone = transformers.MaskFormerSwinConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    # Currently not supported because there's no natten package in conda and no windows support
    elif bb == "NAT":
        backbone = transformers.NatConfig(
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "PVTV2":
        backbone = transformers.PvtV2Config(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "ResNet":
        backbone = transformers.ResNetConfig(
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "RT-DETR-ResNet":
        backbone = transformers.RTDetrResNetConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "Swin":
        backbone = transformers.SwinConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "SwinV2":
        backbone = transformers.Swinv2Config(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif bb == "ViTDet":
        backbone = transformers.VitDetConfig(
            image_size=input_shape,
            num_channels=input_dims,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    else:
        warnings.warn(
            bb + " is not one of the default backbones. Trying to load timm backbone with the requested name.",
            stacklevel=2,
        )
        backbone = transformers.TimmBackboneConfig(
            bb,
            num_channels=input_dims,
        )
    return backbone


class TransformersModel(torch.nn.Module):
    """A custom class that includes data pre- and post-processing to make Transformers models behave like others."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        preprocessor: transformers.BaseImageProcessorFast,
        input_shape: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = preprocessor
        self.input_shape = input_shape

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model."""
        x = batch["x"]
        y = batch.get("y")

        inputs = {"images": x, "input_data_format": "channels_first", "return_tensors": "pt"}
        # Setting up y if processor can process it
        if y is not None:
            inputs["segmentation_maps"] = y
        # Process
        inputs = self.processor(**inputs)
        # Predict
        pred = self.model(**inputs)
        # Get loss
        loss = pred.loss
        # Postprocess
        pred = self.post_process_regression(
            pred,
            target_sizes=[(self.input_shape, self.input_shape)] * x.shape[0],
        )
        pred = torch.stack(pred)
        return pred, loss

    def post_process_regression(self, outputs: Any, target_sizes: Optional[list[tuple]] = None) -> list[torch.Tensor]:
        """
        Adapted from post_process_semantic_segmentation.

        Converts the output of [`SegformerForSemanticSegmentation`] into regression maps. Only supports PyTorch.

        Parameters
        ----------
            outputs ([`SegformerForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns
        -------
            regression: `List[torch.Tensor]` of length `batch_size`, where each item is a regression
             map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")

            regression = []

            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_map = resized_logits[0]
                regression.append(semantic_map)
        else:
            regression = logits
            regression = [regression[i] for i in range(regression.shape[0])]

        return regression


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


class RegressionModels:
    """Regression models base class."""

    model: Union[TorchNNModel, SKLModel]
    model_name: str
    input_shape: int
    input_dims: int
    num_classes: int = 1
    y_nodata: Optional[Union[int, float]]

    def load_model(
        self,
        model_name: str,
        bb: Optional[str],
        weights: Optional[str],
        **kwargs: Any,
    ) -> TorchNNModel:
        """Load a Torch-based regression model."""
        if model_name == "BEiT":
            if weights is not None:
                processor = transformers.BeitImageProcessorFast.from_pretrained(
                    weights,
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
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
            else:
                processor = transformers.BeitImageProcessorFast(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                config = transformers.BeitConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    num_labels=self.num_classes,
                    out_indices=[3, 5, 7, 11],
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.BeitForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape)
        elif model_name == "Data2Vec":
            if weights is not None:
                processor = transformers.AutoImageProcessor.from_pretrained(
                    weights,
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
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
            else:
                processor = transformers.BeitImageProcessorFast(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                config = transformers.Data2VecVisionConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    num_labels=self.num_classes,
                    out_indices=[3, 5, 7, 11],
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.Data2VecVisionForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape)
        elif model_name == "DPT":
            if weights is not None:
                processor = transformers.DPTImageProcessorFast.from_pretrained(
                    weights,
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
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
            else:
                processor = transformers.DPTImageProcessorFast(
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                config = transformers.DPTConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.DPTForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape)
        elif model_name == "MobileNetV2":
            if weights is not None:
                processor = transformers.MobileNetV2ImageProcessorFast.from_pretrained(
                    weights,
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
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
            else:
                processor = transformers.MobileNetV2ImageProcessorFast(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                config = transformers.MobileNetV2Config(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.MobileNetV2ForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape)
        elif model_name == "MobileViT":
            if weights is not None:
                processor = transformers.MobileViTImageProcessorFast.from_pretrained(
                    weights,
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
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
            else:
                processor = transformers.MobileViTImageProcessorFast(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_flip_channel_order=False,
                )
                config = transformers.MobileViTConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.MobileViTForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape)
        elif model_name == "MobileViTV2":
            if weights is not None:
                processor = transformers.MobileViTImageProcessorFast.from_pretrained(
                    weights,
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
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
            else:
                processor = transformers.MobileViTImageProcessorFast(
                    do_resize=False,
                    do_center_crop=False,
                    do_rescale=False,
                    do_flip_channel_order=False,
                )
                config = transformers.MobileViTV2Config(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.MobileViTV2ForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape)
        elif model_name == "SegFormer":
            if weights is not None:
                processor = transformers.SegformerImageProcessorFast.from_pretrained(
                    weights,
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                model = transformers.SegformerForSemanticSegmentation.from_pretrained(
                    weights,
                    ignore_mismatched_sizes=True,
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
            else:
                processor = transformers.SegformerImageProcessorFast(
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                config = transformers.SegformerConfig(
                    image_size=self.input_shape,
                    num_channels=self.input_dims,
                    num_labels=self.num_classes,
                    semantic_loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.SegformerForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape)
        elif model_name == "UperNet":
            if weights is not None:
                processor = transformers.SegformerImageProcessorFast.from_pretrained(
                    weights,
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                config = transformers.AutoConfig.from_pretrained(
                    weights,
                    num_labels=self.num_classes,
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
            else:
                processor = transformers.SegformerImageProcessorFast(
                    do_resize=False,
                    do_normalize=False,
                    do_rescale=False,
                )
                if bb is None:
                    bb = "Swin"
                backbone = load_backbone(bb, self.input_shape, self.input_dims)
                # ResNet auxiliary_in_channels=1024
                config = transformers.UperNetConfig(
                    backbone_config=backbone,
                    num_labels=self.num_classes,
                    loss_ignore_index=self.y_nodata,
                    **kwargs,
                )
                model = transformers.UperNetForSemanticSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape)
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
            model = torchgeo.models.FarSeg(
                backbone=bb if bb is not None else "resnet50",
                classes=self.num_classes,
                backbone_pretrained=weights is not None,
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
        return CustomModel(model)

    def load_sklearn_model(
        self,
        model_name: str,
        bb: Optional[str],
        max_depth: Optional[int] = 6,
        **kwargs: Any,
    ) -> SKLModel:
        """Load a Sklearn-based regression model."""
        if model_name == "Linear Regression":
            model = LinearRegression(n_jobs=-1, **kwargs)
        elif model_name == "Ridge":
            model = Ridge(**kwargs)
        elif model_name == "Bayesian Ridge":
            model = BayesianRidge(verbose=True, **kwargs)
        elif model_name == "Lasso":
            model = Lasso(warm_start=True, **kwargs)
        elif model_name == "Multitask Lasso":
            model = MultiTaskLasso(warm_start=True, **kwargs)
        elif model_name == "Lars":
            model = Lars(verbose=True, **kwargs)
        elif model_name == "LassoLars":
            model = LassoLars(verbose=True, **kwargs)
        elif model_name == "LassoLarsIC":
            model = LassoLarsIC(verbose=True, **kwargs)
        elif model_name == "ElasticNet":
            model = ElasticNet(**kwargs)
        elif model_name == "Multitask ElasticNet":
            model = MultiTaskElasticNet(warm_start=True, **kwargs)
        elif model_name == "Orthogonal Matching Pursuit":
            model = OrthogonalMatchingPursuit(**kwargs)
        elif model_name == "ARD":
            model = ARDRegression(verbose=True, **kwargs)
        elif model_name == "Huber":
            model = HuberRegressor(warm_start=True, **kwargs)
        elif model_name == "RANSAC":
            model = RANSACRegressor(**kwargs)
        elif model_name == "Theil-Sen":
            model = TheilSenRegressor(n_jobs=-1, verbose=True, **kwargs)
        elif model_name == "Gamma":
            if bb == "lbfgs" or bb is None:
                model = GammaRegressor(solver="lbfgs", warm_start=True, verbose=1000, **kwargs)
            elif bb == "newton-cholesky":
                model = GammaRegressor(solver="newton-cholesky", warm_start=True, verbose=1000, **kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "Poisson":
            if bb == "lbfgs" or bb is None:
                model = PoissonRegressor(solver="lbfgs", warm_start=True, verbose=1000, **kwargs)
            elif bb == "newton-cholesky":
                model = PoissonRegressor(solver="newton-cholesky", warm_start=True, verbose=1000, **kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "Tweedie":
            if bb == "lbfgs" or bb is None:
                model = TweedieRegressor(solver="lbfgs", warm_start=True, verbose=1000, **kwargs)
            elif bb == "newton-cholesky":
                model = TweedieRegressor(solver="newton-cholesky", warm_start=True, verbose=1000, **kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "SGD":
            if bb == "squared_error" or bb is None:
                model = SGDRegressor(loss="squared_error", warm_start=True, verbose=1000, **kwargs)
            elif bb == "huber":
                model = SGDRegressor(loss="huber", warm_start=True, verbose=1000, **kwargs)
            elif bb == "epsilon_insensitive":
                model = SGDRegressor(loss="epsilon_insensitive", warm_start=True, verbose=1000, **kwargs)
            elif bb == "squared_epsilon_insensitive":
                model = SGDRegressor(loss="squared_epsilon_insensitive", warm_start=True, verbose=1000, **kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "Nearest Neighbors":
            model = KNeighborsRegressor(n_jobs=-1, **kwargs)
        elif model_name == "Radius Neighbors":
            model = RadiusNeighborsRegressor(n_jobs=-1, **kwargs)
        elif model_name == "SVM":
            if bb == "rbf" or bb is None:
                model = SVR(kernel="rbf", verbose=True, **kwargs)
            elif bb == "linear":
                model = LinearSVR(verbose=1, **kwargs)
            elif bb == "poly":
                model = SVR(kernel="poly", verbose=True, **kwargs)
            elif bb == "sigmoid":
                model = SVR(kernel="sigmoid", verbose=True, **kwargs)
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "Gaussian Process":
            model = GaussianProcessRegressor(**kwargs)
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(**kwargs)
        elif model_name == "Extra Tree":
            model = ExtraTreeRegressor(**kwargs)
        elif model_name == "Random Forest":
            # max_depth is by default set to 6, because it is unlimited by default and the training could be very slow.
            # To train with unlimited tree depth set max_depth = None
            model = RandomForestRegressor(max_depth=max_depth, n_jobs=-1, warm_start=True, verbose=1, **kwargs)
        elif model_name == "Extra Trees":
            model = ExtraTreesRegressor(max_depth=max_depth, n_jobs=-1, warm_start=True, verbose=1, **kwargs)
        elif model_name == "AdaBoost":
            model = AdaBoostRegressor(**kwargs)
        elif model_name == "Gradient Boosting":
            model = HistGradientBoostingRegressor(warm_start=True, validation_fraction=None, verbose=1000, **kwargs)
        elif model_name == "Multilayer Perceptron":
            if bb == "adam" or bb is None:
                model = MLPRegressor(
                    solver="adam",
                    warm_start=True,
                    verbose=True,
                    **kwargs,
                )
            elif bb == "sgd":
                model = MLPRegressor(
                    solver="sgd",
                    warm_start=True,
                    verbose=True,
                    **kwargs,
                )
            elif bb == "lbfgs":
                model = MLPRegressor(
                    solver="lbfgs",
                    warm_start=True,
                    verbose=True,
                    **kwargs,
                )
            else:
                raise ValueError("Unknown backbone " + bb)
        elif model_name == "XGBoost":
            model = xgb.XGBRegressor(tree_method="hist", verbosity=3, n_jobs=-1, **kwargs)
        elif model_name == "XGB Random Forest":
            model = xgb.XGBRFRegressor(tree_method="hist", verbosity=3, n_jobs=-1, **kwargs)
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
    "Data2Vec",
    "DETR",
    "DPT",
    "MobileNetV2",
    "MobileViT",
    "MobileViTV2",
    "SegFormer",
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
    "Linear Regression",
    "Ridge",
    "Bayesian Ridge",
    "Lasso",
    "Multitask Lasso",
    "Lars",
    "LassoLars",
    "LassoLarsIC",
    "ElasticNet",
    "Multitask ElasticNet",
    "Orthogonal Matching Pursuit",
    "ARD",
    "Huber",
    "RANSAC",
    "Theil-Sen",
    "Gamma",
    "Poisson",
    "Tweedie",
    "SGD",
    "Nearest Neighbors",
    "Radius Neighbors",
    "SVM",
    "Gaussian Process",
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
