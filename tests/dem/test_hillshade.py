"""Test hillshade calculation."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor.dem import hillshade


def test_hillshade(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test hillshade calculation."""
    hillshade_file = tmp_path / "hillshade_fast.tif"
    hillshade(dem_toy_data, output_path=hillshade_file)


def test_hillshade_custom(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test hillshade calculation with custom parameters."""
    hillshade_file = tmp_path / "hillshade_norm_fast.tif"
    hillshade(dem_toy_data, output_path=hillshade_file, azimuth=90, angle_altitude=5)


@pytest.mark.slow
def test_hillshade_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test hillshade calculation."""
    hillshade_file = tmp_path / "hillshade_slow.tif"
    item = load_dem_grand_canyon
    hillshade(item, output_path=hillshade_file)


@pytest.mark.slow
def test_hillshade_norm_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test normalized hillshade calculation with custom parameters."""
    hillshade_file = tmp_path / "hillshade_norm_slow.tif"
    item = load_dem_grand_canyon
    hillshade(item, output_path=hillshade_file, azimuth=90, angle_altitude=5)
