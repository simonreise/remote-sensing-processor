"""DEM processing module."""

from remote_sensing_processor.dem.aspect import aspect
from remote_sensing_processor.dem.curvature import curvature
from remote_sensing_processor.dem.hillshade import hillshade
from remote_sensing_processor.dem.slope import slope


__all__ = ["aspect", "curvature", "hillshade", "slope"]
