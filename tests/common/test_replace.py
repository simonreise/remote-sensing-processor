"""Test replace values."""

from typing import Union

from pathlib import Path

import pytest

import rioxarray as rxr

from pystac import Item

from remote_sensing_processor import replace_nodata, replace_value


def test_replace_single(tmp_path: Path, esa_worldcover_toy_data: Path) -> None:
    """Test replacing single value."""
    replace_file = tmp_path / "replace_fast_one.tif"
    replace_value(esa_worldcover_toy_data, output_path=replace_file, old=20, new=30)
    assert_value_not_in_file(replace_file, value=20)


def test_replace_multiple(tmp_path: Path, esa_worldcover_toy_data: Path) -> None:
    """Test replacing multiple values."""
    replace_file = tmp_path / "replace_fast_many.tif"
    replace_value(
        esa_worldcover_toy_data,
        output_path=replace_file,
        values={10: 1, 20: 1, 30: 2, 40: 2, 50: 4, 60: 4, 70: 3, 80: 2, 90: 1, 100: 2},
    )
    assert_value_not_in_file(replace_file, value=10)


def test_replace_nodata(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test replacing nodata value."""
    replace_file = tmp_path / "replace_fast_nodata.tif"
    replace_nodata(dem_toy_data, output_path=replace_file, new=0, old=-9999)
    assert_nodata(replace_file, value=0)


@pytest.mark.slow
def test_replace_single_slow(tmp_path: Path, load_lulc_roma: Item) -> None:
    """Test replacing single value."""
    replace_file = tmp_path / "replace_slow_one.tif"
    item = load_lulc_roma
    replace_value(item, output_path=replace_file, old=20, new=30)
    assert_value_not_in_file(replace_file, value=20)


@pytest.mark.slow
def test_replace_multiple_slow(tmp_path: Path, load_lulc_roma: Item) -> None:
    """Test replacing multiple values."""
    replace_file = tmp_path / "replace_slow_many.tif"
    item = load_lulc_roma
    replace_value(
        item,
        output_path=replace_file,
        values={10: 1, 20: 1, 30: 2, 40: 2, 50: 4, 60: 4, 70: 3, 80: 2, 90: 1, 100: 2},
    )
    assert_value_not_in_file(replace_file, value=10)


@pytest.mark.slow
def test_replace_nodata_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test replacing nodata value."""
    replace_file = tmp_path / "replace_slow_nodata.tif"
    item = load_dem_grand_canyon
    replace_nodata(item, output_path=replace_file, new=0, old=-9999)
    assert_nodata(replace_file, value=0)


def assert_value_not_in_file(file: Path, value: Union[int, float]) -> None:
    """Assert if value successfully replaced."""
    with rxr.open_rasterio(file) as raster:
        assert value not in raster.values


def assert_nodata(file: Path, value: Union[int, float]) -> None:
    """Assert nodata value."""
    with rxr.open_rasterio(file) as raster:
        assert raster.rio.nodata == value
