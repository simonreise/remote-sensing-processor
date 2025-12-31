"""Remote Sensing Processor."""

__version__ = "0.2.2"


from remote_sensing_processor.common.process import process

from remote_sensing_processor.common.replace import replace_value, replace_nodata

from remote_sensing_processor.common.rasterize import rasterize

from remote_sensing_processor.common.match_hist import match_hist

from remote_sensing_processor.common.clip_values import clip_values

# A function that get dask client from a cluster, or creates local cluster.
# Will be useful if functions will support computation on dask clusters.
# TODO : make everything support computation on dask or ray clusters
# from remote_sensing_processor.common.dask import get_client

from remote_sensing_processor.common.normalize import normalize, denormalize, get_normalization_params

from remote_sensing_processor.indices.vegetation_index import calculate_index

from remote_sensing_processor import dem

from remote_sensing_processor.imagery.landsat.landsat import landsat

from remote_sensing_processor.imagery.sentinel2.sentinel2 import sentinel2

from remote_sensing_processor.mosaic.mosaic import mosaic

from remote_sensing_processor.segmentation import (
    semantic,
    # instance,
    # panoptic,
    regression,
    # unsupervised,
    # object_detection,
)

import warnings
import rasterio.errors

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

__all__ = [
    "process",
    "replace_value",
    "replace_nodata",
    "rasterize",
    "match_hist",
    "clip_values",
    "normalize",
    "denormalize",
    "get_normalization_params",
    "calculate_index",
    "dem",
    "landsat",
    "sentinel2",
    "mosaic",
    "semantic",
    # "instance",
    # "panoptic",
    "regression",
    # "unsupervised",
    # "object_detection",
]
