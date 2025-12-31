"""Regression losses."""

from typing import Optional

import torch

from remote_sensing_processor.common.types import TorchLoss


losses = [
    "mae",
    "mse",
]


def setup_loss(loss: Optional[str]) -> TorchLoss:
    """Setup loss function."""
    if loss is None:
        loss = "mse"
    if isinstance(loss, str):
        if loss == "mse" or loss is None:
            loss_fn = torch.nn.MSELoss()
        elif loss == "mae":
            loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError("Unknown loss function " + loss)
    else:
        loss_fn = loss

    return loss_fn
