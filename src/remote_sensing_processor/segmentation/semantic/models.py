"""Semantic segmentation models."""

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
    """A custom class that includes data pre- and post-processing for Transformers models."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        preprocessor: Union[transformers.BaseImageProcessorFast, transformers.OneFormerProcessor],
        input_shape: int,
        y_nodata: Optional[int] = None,
        postprocessor: Optional[str] = "general",
        detr: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = preprocessor
        self.postprocessor = postprocessor
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
            # noinspection PyTypeChecker
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
        if self.postprocessor == "general":
            pred = self.post_process_semantic_segmentation(
                pred,
                target_sizes=[(self.input_shape, self.input_shape)] * x.shape[0],
            )
        elif self.postprocessor == "conditional_detr":
            pred = self.post_process_semantic_segmentation_conditional_detr(
                pred,
                target_sizes=[(self.input_shape, self.input_shape)] * x.shape[0],
            )
        elif self.postprocessor == "detr":
            pred = self.post_process_semantic_segmentation_detr(
                pred,
                target_sizes=[(self.input_shape, self.input_shape)] * x.shape[0],
            )
        elif self.postprocessor == "maskformer":
            pred = self.post_process_semantic_segmentation_maskformer(
                pred,
                target_sizes=[(self.input_shape, self.input_shape)] * x.shape[0],
            )
        elif self.postprocessor == "eomt":
            pred = self.post_process_semantic_segmentation_eomt(
                pred,
                target_sizes=[(self.input_shape, self.input_shape)] * x.shape[0],
            )
        pred = torch.stack(pred)
        return pred, loss

    def prepare_detr_annotation(self, sem_seg: torch.Tensor, y_nodata: Optional[int], input_shape: int) -> dict:
        """Function that converts semantic segmentation maps to DETR annotations."""
        from transformers.models.detr.image_processing_detr_fast import DetrImageProcessorFast, masks_to_boxes

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
        return DetrImageProcessorFast.normalize_annotation(None, annotation, (input_shape, input_shape))

    def post_process_semantic_segmentation(
        self,
        outputs: Any,
        target_sizes: Optional[list[tuple]] = None,
    ) -> list[torch.Tensor]:
        """
        Converts the output of [`SegFormerForSemanticSegmentation`] into semantic segmentation maps.

        Only supports PyTorch.

        Parameters
        ----------
            outputs ([`MobileNetV2ForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns
        -------
            semantic_segmentation: `list[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")

            semantic_segmentation = []

            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_map = resized_logits[0]
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = logits
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_semantic_segmentation_conditional_detr(
        self,
        outputs: Any,
        target_sizes: Optional[list[tuple[int, int]]] = None,
    ) -> list[torch.Tensor]:
        """
        Converts the output of [`ConditionalDetrForSegmentation`] to semantic segmentation maps. Only supports PyTorch.

        Parameters
        ----------
            outputs ([`ConditionalDetrForSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                A list of tuples (`tuple[int, int]`) containing the target size (height, width) of each image in the
                batch. If unset, predictions will not be resized.

        Returns
        -------
            `list[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits",
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_map = resized_logits[0]
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_semantic_segmentation_detr(
        self,
        outputs: Any,
        target_sizes: Optional[list[tuple[int, int]]] = None,
    ) -> list[torch.Tensor]:
        """
        Converts the output of [`DetrForSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Parameters
        ----------
            outputs ([`DetrForSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                A list of tuples (`tuple[int, int]`) containing the target size (height, width) of each image in the
                batch. If unset, predictions will not be resized.

        Returns
        -------
            `list[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits",
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_map = resized_logits[0]
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_semantic_segmentation_maskformer(
        self,
        outputs: Any,
        target_sizes: Optional[list[tuple[int, int]]] = None,
    ) -> list[torch.Tensor]:
        """
        Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps.

        Only supports PyTorch.

        Parameters
        ----------
            outputs ([`MaskFormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`tuple[int, int]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.

        Returns
        -------
            `list[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_map = resized_logits[0]
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_semantic_segmentation_eomt(
        self,
        outputs: Any,
        target_sizes: list[tuple[int, int]],
        size: Optional[dict[str, int]] = None,
    ) -> list[torch.Tensor]:
        """Post-processes model outputs into final semantic segmentation prediction."""
        size = size if size is not None else self.processor.size

        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        patch_offsets = outputs.patch_offsets

        output_size = (size["shortest_edge"], size["longest_edge"] or size["shortest_edge"])
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=output_size,
            mode="bilinear",
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        if patch_offsets:
            output_logits = self.merge_image_patches(segmentation_logits, patch_offsets, target_sizes, size)
        else:
            output_logits = []

            for idx in range(len(segmentation_logits)):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation_logits[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                output_logits.append(resized_logits[0])

        return output_logits


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
        if model_name == "BEiT":
            if weights is not None:
                processor = transformers.BeitImageProcessorFast.from_pretrained(
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
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "ConditionalDETR":
            if weights is not None:
                processor = transformers.ConditionalDetrImageProcessorFast.from_pretrained(
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
                    num_labels=self.num_classes,
                    **kwargs,
                )
            else:
                processor = transformers.ConditionalDetrImageProcessorFast(
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                if bb is None:
                    config = transformers.ConditionalDetrConfig(
                        num_channels=self.input_dims,
                        num_labels=self.num_classes,
                        **kwargs,
                    )
                else:
                    backbone = load_backbone(bb, input_shape=self.input_shape, input_dims=self.input_dims)
                    config = transformers.ConditionalDetrConfig(
                        backbone_config=backbone,
                        backbone=None,
                        use_timm_backbone=False,
                        num_channels=self.input_dims,
                        num_labels=self.num_classes,
                        **kwargs,
                    )
                model = transformers.ConditionalDetrForSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata, "conditional_detr", True)
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
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "DETR":
            # Processor not working with segmentation maps
            if weights is not None:
                processor = transformers.DetrImageProcessorFast.from_pretrained(
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
                    num_labels=self.num_classes,
                    **kwargs,
                )
            else:
                processor = transformers.DetrImageProcessorFast(
                    do_resize=False,
                    do_pad=False,
                    do_rescale=False,
                    do_normalize=False,
                )
                if bb is None:
                    config = transformers.DetrConfig(
                        num_channels=self.input_dims,
                        num_labels=self.num_classes,
                        **kwargs,
                    )
                else:
                    backbone = load_backbone(bb, input_shape=self.input_shape, input_dims=self.input_dims)
                    config = transformers.DetrConfig(
                        backbone_config=backbone,
                        backbone=None,
                        use_timm_backbone=False,
                        num_channels=self.input_dims,
                        num_labels=self.num_classes,
                        **kwargs,
                    )
                model = transformers.DetrForSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata, "detr", True)
        elif model_name == "DPT":
            if weights is not None:
                processor = transformers.DPTImageProcessorFast.from_pretrained(
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
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "EoMT":
            if weights is not None:
                processor = transformers.EomtImageProcessorFast.from_pretrained(
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
                    num_labels=self.num_classes,
                    ignore_value=self.y_nodata,
                    **kwargs,
                )
            else:
                processor = transformers.EomtImageProcessorFast(
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
                    num_labels=self.num_classes,
                    ignore_value=self.y_nodata,
                    **kwargs,
                )
                model = transformers.EomtForUniversalSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata, "eomt")
        elif model_name == "Mask2Former":
            if weights is not None:
                processor = transformers.Mask2FormerImageProcessorFast.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    num_labels=self.num_classes,
                    ignore_index=self.y_nodata,
                )
                config = transformers.AutoConfig.from_pretrained(
                    weights,
                    num_labels=self.num_classes,
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
            else:
                processor = transformers.Mask2FormerImageProcessorFast(
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    num_labels=self.num_classes,
                    ignore_index=self.y_nodata,
                )
                if bb is None:
                    bb = "Swin"
                backbone = load_backbone(bb, self.input_shape, self.input_dims)
                config = transformers.Mask2FormerConfig(
                    backbone_config=backbone,
                    num_labels=self.num_classes,
                    ignore_value=self.y_nodata,
                    **kwargs,
                )
                model = transformers.Mask2FormerForUniversalSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata, "maskformer")
        elif model_name == "MaskFormer":
            if weights is not None:
                processor = transformers.MaskFormerImageProcessorFast.from_pretrained(
                    weights,
                    use_fast=True,
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    num_labels=self.num_classes,
                    ignore_index=self.y_nodata,
                )
                config = transformers.AutoConfig.from_pretrained(
                    weights,
                    num_labels=self.num_classes,
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
            else:
                processor = transformers.MaskFormerImageProcessorFast(
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    num_labels=self.num_classes,
                    ignore_index=self.y_nodata,
                )
                if bb is None:
                    bb = "Swin"
                backbone = load_backbone(bb, self.input_shape, self.input_dims)
                config = transformers.MaskFormerConfig(
                    backbone_config=backbone,
                    num_labels=self.num_classes,
                    **kwargs,
                )
                model = transformers.MaskFormerForInstanceSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata, "maskformer")
        elif model_name == "MobileNetV2":
            if weights is not None:
                processor = transformers.MobileNetV2ImageProcessorFast.from_pretrained(
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
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "MobileViT":
            if weights is not None:
                processor = transformers.MobileViTImageProcessorFast.from_pretrained(
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
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "MobileViTV2":
            if weights is not None:
                processor = transformers.MobileViTImageProcessorFast.from_pretrained(
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
                    num_labels=self.num_classes,
                    ignore_index=self.y_nodata,
                    repo_path=Path(temp.name).parent.as_posix(),
                    class_info_file=Path(temp.name).name,
                    num_text=150,
                )
                config = transformers.AutoConfig.from_pretrained(
                    weights,
                    is_training=True,
                    num_labels=self.num_classes,
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
            else:
                processor = transformers.OneFormerImageProcessorFast(
                    do_resize=False,
                    do_rescale=False,
                    do_normalize=False,
                    num_labels=self.num_classes,
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
                    num_labels=self.num_classes,
                    ignore_value=self.y_nodata,
                    is_training=True,
                    num_queries=150,
                    text_encoder_n_ctx=16,
                    **kwargs,
                )
                model = transformers.OneFormerForUniversalSegmentation(config)
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata, "maskformer")
            Path(temp.name).unlink()
        elif model_name == "SegFormer":
            if weights is not None:
                processor = transformers.SegformerImageProcessorFast.from_pretrained(
                    weights,
                    use_fast=True,
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
            model = TransformersModel(model, processor, self.input_shape, self.y_nodata)
        elif model_name == "UperNet":
            if weights is not None:
                processor = transformers.SegformerImageProcessorFast.from_pretrained(
                    weights,
                    use_fast=True,
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
                config = transformers.UperNetConfig(
                    backbone_config=backbone,
                    num_labels=self.num_classes,
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
    "Mask2Former",
    "MaskFormer",
    "MobileNetV2",
    "MobileViT",
    "MobileViTV2",
    "OneFormer",
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
