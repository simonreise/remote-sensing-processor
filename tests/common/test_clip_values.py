"""Test clipping values."""

from typing import Optional, Union

from pathlib import Path

import pytest

import rioxarray as rxr

from pystac import Item

from remote_sensing_processor import clip_values


def test_clip_values_both(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test clipping values by min and max values."""
    clip_file = tmp_path / "clip_fast_both.tif"
    clip_values(dem_toy_data, output_path=clip_file, minimum=4000, maximum=5500)
    assert_min_max(clip_file, minimum=4000, maximum=5500)


def test_clip_values_min(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test clipping values by only min value."""
    clip_file = tmp_path / "clip_fast_min.tif"
    clip_values(dem_toy_data, output_path=clip_file, minimum=4000)
    assert_min_max(clip_file, minimum=4000)


def test_clip_values_max(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test clipping values by only max value."""
    clip_file = tmp_path / "clip_fast_max.tif"
    clip_values(dem_toy_data, output_path=clip_file, maximum=5500)
    assert_min_max(clip_file, maximum=5500)


@pytest.mark.slow
def test_clip_values_both_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test clipping values by min and max values."""
    clip_file = tmp_path / "clip_fast_both.tif"
    item = load_dem_grand_canyon
    clip_values(item, output_path=clip_file, minimum=800, maximum=2400)
    assert_min_max(clip_file, minimum=800, maximum=2400)


@pytest.mark.slow
def test_clip_values_min_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test clipping values by only min value."""
    clip_file = tmp_path / "clip_fast_min.tif"
    item = load_dem_grand_canyon
    clip_values(item, output_path=clip_file, minimum=800)
    assert_min_max(clip_file, minimum=800)


@pytest.mark.slow
def test_clip_values_max_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test clipping values by only max value."""
    clip_file = tmp_path / "clip_fast_max.tif"
    item = load_dem_grand_canyon
    clip_values(item, output_path=clip_file, maximum=2400)
    assert_min_max(clip_file, maximum=2400)


def assert_min_max(file: Path, minimum: Optional[Union[int]] = None, maximum: Optional[Union[int]] = None) -> None:
    """Assert if clipping is successful."""
    with rxr.open_rasterio(file) as raster:
        if minimum is not None:
            assert raster.min() >= minimum
        if maximum is not None:
            assert raster.max() <= maximum
