"""Test process rasters."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor import process


def test_process_fillna_crs(
    tmp_path: Path,
    esa_worldcover_toy_data: Path,
    esa_worldcover_toy_data_clip: Path,
) -> None:
    """Test processing rasters with filling nodata and reprojection."""
    process_file = tmp_path / "process_fast_fillna.tif"
    process(
        esa_worldcover_toy_data,
        output_path=process_file,
        clip=esa_worldcover_toy_data_clip,
        crs=3857,
        fill_nodata=True,
        fill_distance=10,
    )


def test_process_reference_nostac(
    tmp_path: Path,
    esa_worldcover_toy_data: Path,
    esa_worldcover_toy_data_clip: Path,
) -> None:
    """Test processing rasters with matching to another raster and not writing STAC."""
    process_file = tmp_path / "process_fast_fillna.tif"
    out = process(
        esa_worldcover_toy_data,
        output_path=process_file,
        clip=esa_worldcover_toy_data_clip,
        reference_raster=esa_worldcover_toy_data,
        resample="nearest",
        write_stac=False,
    )
    assert ".json" not in out.suffixes


@pytest.mark.slow
def test_process_slow(tmp_path: Path, load_lulc_roma: Item, roma_clip: Path) -> None:
    """Test processing rasters."""
    process_file = tmp_path / "process_slow.tif"
    item = load_lulc_roma
    process(item, output_path=process_file, clip=roma_clip, fill_nodata=True, fill_distance=10)
