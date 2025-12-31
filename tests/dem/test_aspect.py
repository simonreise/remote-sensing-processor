"""Test aspect calculation."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor.dem import aspect


def test_aspect(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test aspect calculation."""
    aspect_file = tmp_path / "aspect_fast.tif"
    aspect(dem_toy_data, output_path=aspect_file)


def test_aspect_norm(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test normalized aspect calculation."""
    aspect_file = tmp_path / "aspect_norm_fast.tif"
    aspect(dem_toy_data, output_path=aspect_file, normalize=True)


@pytest.mark.slow
def test_aspect_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test aspect calculation."""
    aspect_file = tmp_path / "aspect_slow.tif"
    item = load_dem_grand_canyon
    aspect(item, output_path=aspect_file)


@pytest.mark.slow
def test_aspect_norm_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test normalized aspect calculation."""
    aspect_file = tmp_path / "aspect_norm_slow.tif"
    item = load_dem_grand_canyon
    aspect(item, output_path=aspect_file, normalize=True)
