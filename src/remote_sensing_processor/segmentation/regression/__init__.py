"""Regression module."""

from remote_sensing_processor.segmentation.regression.segmentation import train, test
from remote_sensing_processor.segmentation.regression.tiles import generate_tiles
from remote_sensing_processor.segmentation.regression.mapping import generate_map
from remote_sensing_processor.segmentation.regression.analysis import band_importance

__all__ = ["generate_tiles", "train", "test", "generate_map", "band_importance"]
