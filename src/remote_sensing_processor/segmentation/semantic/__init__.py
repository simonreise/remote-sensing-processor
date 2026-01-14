"""Semantic segmentation module."""

from remote_sensing_processor.segmentation.semantic.segmentation import train, test
from remote_sensing_processor.segmentation.semantic.tiles import generate_tiles
from remote_sensing_processor.segmentation.semantic.mapping import generate_map
from remote_sensing_processor.segmentation.semantic.analysis import band_importance, confusion_matrix


__all__ = ["generate_tiles", "train", "test", "generate_map", "band_importance", "confusion_matrix"]
