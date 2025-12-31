"""Test curvature calculation."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor.dem import curvature


def test_curvature(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test curvature calculation."""
    curvature_file = tmp_path / "curvature_fast.tif"
    curvature(dem_toy_data, output_path=curvature_file)


def test_curvature_norm(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test normalized curvature calculation."""
    curvature_file = tmp_path / "curvature_norm_fast.tif"
    curvature(dem_toy_data, output_path=curvature_file, normalize=True)


@pytest.mark.slow
def test_curvature_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test curvature calculation."""
    curvature_file = tmp_path / "curvature_slow.tif"
    item = load_dem_grand_canyon
    curvature(item, output_path=curvature_file)


@pytest.mark.slow
def test_curvature_norm_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test normalized curvature calculation."""
    curvature_file = tmp_path / "curvature_norm_slow.tif"
    item = load_dem_grand_canyon
    curvature(item, output_path=curvature_file, normalize=True)
