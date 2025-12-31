"""Test slope calculation."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor.dem import slope


def test_slope(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test slope calculation."""
    slope_file = tmp_path / "slope_fast.tif"
    slope(dem_toy_data, output_path=slope_file)


def test_slope_norm(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test normalized slope calculation."""
    slope_file = tmp_path / "slope_norm_fast.tif"
    slope(dem_toy_data, output_path=slope_file, normalize=True)


@pytest.mark.slow
def test_slope_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test slope calculation."""
    slope_file = tmp_path / "slope_slow.tif"
    item = load_dem_grand_canyon
    slope(item, output_path=slope_file)


@pytest.mark.slow
def test_slope_norm_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test normalized slope calculation."""
    slope_file = tmp_path / "slope_norm_slow.tif"
    item = load_dem_grand_canyon
    slope(item, output_path=slope_file, normalize=True)
