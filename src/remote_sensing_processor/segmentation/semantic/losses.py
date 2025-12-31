"""Semantic segmentation losses."""

from typing import Optional

import segmentation_models_pytorch
import torch

from remote_sensing_processor.common.types import TorchLoss


losses = [
    "cross_entropy",
    "nll",
    "jaccard",
    "dice",
    "tversky",
    "focal",
    "lovasz",
]


def setup_loss(loss: Optional[str], y_nodata: Optional[int] = None) -> TorchLoss:
    """Setup loss function."""
    if loss is None:
        loss = "cross_entropy"
    if isinstance(loss, str):
        if loss == "cross_entropy" or loss is None:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=y_nodata)
        elif loss == "nll":
            loss_fn = torch.nn.NLLLoss(ignore_index=y_nodata)
        elif loss == "jaccard":
            loss_fn = segmentation_models_pytorch.losses.JaccardLoss(mode="multiclass", ignore_index=y_nodata)
        elif loss == "dice":
            loss_fn = segmentation_models_pytorch.losses.DiceLoss(mode="multiclass", ignore_index=y_nodata)
        elif loss == "tversky":
            loss_fn = segmentation_models_pytorch.losses.TverskyLoss(mode="multiclass", ignore_index=y_nodata)
        elif loss == "focal":
            loss_fn = segmentation_models_pytorch.losses.FocalLoss(mode="multiclass", ignore_index=y_nodata)
        elif loss == "lovasz":
            loss_fn = segmentation_models_pytorch.losses.LovaszLoss(mode="multiclass", ignore_index=y_nodata)
        else:
            raise ValueError("Unknown loss function " + loss)
    else:
        loss_fn = loss

    return loss_fn
