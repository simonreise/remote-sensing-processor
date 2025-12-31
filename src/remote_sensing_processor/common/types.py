"""Custom RSP types."""

from pydantic import (
    AfterValidator,
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    InstanceOf,
    StringConstraints,
    TypeAdapter,
)
from typing import Annotated, Any, Literal, Optional, TypeVar, Union

from pathlib import Path

import numpy as np

import rasterio.crs
import rasterio.enums

from pystac import Item

import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchmetrics.metric import Metric
from torchvision.transforms.v2 import Transform

from sklearn.base import BaseEstimator

from remote_sensing_processor.common.dataset import validate


T = TypeVar("T")


def single_item_to_list(v: Union[T, list[T]]) -> list[T]:
    """Allow a single item or a list, but always returns a list."""
    if isinstance(v, list):
        return v
    return [v]


SingleOrList = Annotated[Union[T, list[T]], AfterValidator(single_item_to_list)]

ListOfStr = SingleOrList[str]
ListOfDict = SingleOrList[dict[str, Any]]


def check_if_file(path: Path) -> Path:
    """Custom is file check with resolve."""
    if path.is_file():
        return path.resolve()
    raise ValueError(str(path) + " is not a file")


def check_if_dir(path: Path) -> Path:
    """Custom is dir check with resolve."""
    if path.is_dir():
        return path.resolve()
    raise ValueError(str(path) + " is not a file")


def check_if_exists(path: Path) -> Path:
    """Custom exists check.

    Default NewPath throws error if the parent directory does not exist, and we don't want it
    """
    if not path.exists():
        return path.resolve()
    raise ValueError(str(path) + " exists")


FilePath = Annotated[
    Union[FilePath, str],
    AfterValidator(lambda x: Path(x)),
    AfterValidator(check_if_file),
]

DirectoryPath = Annotated[
    Union[DirectoryPath, str],
    AfterValidator(lambda x: Path(x)),
    AfterValidator(check_if_dir),
]

NewPath = Annotated[
    Union[Path, str],
    AfterValidator(lambda x: Path(x)),
    AfterValidator(check_if_exists),
]

ListOfPath = SingleOrList[Union[FilePath, DirectoryPath]]


def validate_rspds_ext(file: Path) -> Path:
    """Check if file suffix is .rspds."""
    if ".rspds" not in file.suffixes:
        raise ValueError("output must be saved in .rspds format")
    return file


NewRSPDS = Annotated[
    NewPath,
    AfterValidator(lambda x: x.resolve()),
    AfterValidator(lambda x: validate_rspds_ext(x)),
]

LoadRSPDS = Annotated[
    DirectoryPath,
    AfterValidator(lambda x: x.resolve()),
    AfterValidator(lambda x: validate_rspds_ext(x)),
]


def validate_dtype(dtype: str) -> np.dtype:
    """Validate input and convert to a numpy.dtype object."""
    try:
        return np.dtype(dtype)
    except Exception as e:
        raise ValueError(f"Invalid dtype input: {dtype}") from e


DType = Annotated[
    str,
    AfterValidator(validate_dtype),
]

Temperature = Annotated[
    Literal["k", "c", "K", "C"],
    StringConstraints(to_lower=True),
]

Percent = Annotated[int, Field(ge=0, le=100)]


def validate_crs(value: Any) -> rasterio.crs.CRS:
    """Validate input and convert to a rasterio.crs.CRS object."""
    if isinstance(value, rasterio.crs.CRS):
        return value
    try:
        return rasterio.crs.CRS.from_user_input(value)
    except Exception as e:
        raise ValueError(f"Invalid CRS input: {value}") from e


CRS = Annotated[
    Any,
    AfterValidator(validate_crs),
]
crs_adapter = TypeAdapter(CRS)


def validate_pystac(value: Any) -> Item:
    """Validate input and convert to a pystac.Item object."""
    if isinstance(value, Item):
        validate(value)
        return value
    if isinstance(value, dict):
        value = Item.from_dict(value)
        validate(value)
        return value
    raise ValueError(f"Invalid value for pystac.Item: {value}")


PystacItem = Annotated[
    Union[InstanceOf[Item], dict],
    AfterValidator(validate_pystac),
]
ListOfPystacItem = SingleOrList[PystacItem]

TorchNNModel = InstanceOf[nn.Module]

SKLModel = InstanceOf[BaseEstimator]

TorchLoss = InstanceOf[_Loss]

TorchTransform = InstanceOf[Transform]

TorchMetric = InstanceOf[Metric]


class MetricDict(BaseModel):
    """Basic metric class."""

    name: str
    log: Optional[Literal["epoch", "step", "verbose"]] = "step"
    metric: Optional[TorchMetric] = None


ListOfMetrics = SingleOrList[MetricDict]
