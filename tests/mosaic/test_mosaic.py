"""Test raster mosaic creation."""

from pathlib import Path

import pytest

from remote_sensing_processor import mosaic, process


def test_mosaic_worldcover(
    tmp_path: Path,
    esa_worldcover_toy_data: Path,
    esa_worldcover_toy_data_clip1: Path,
    esa_worldcover_toy_data_clip2: Path,
) -> None:
    """Test mosaic."""
    clipped1 = tmp_path / "esa_clipped1.tif"
    clipped1 = process(esa_worldcover_toy_data, output_path=clipped1, clip=esa_worldcover_toy_data_clip1)
    clipped2 = tmp_path / "esa_clipped2.tif"
    clipped2 = process(esa_worldcover_toy_data, output_path=clipped2, clip=esa_worldcover_toy_data_clip2)
    mosaic([clipped1, clipped2], output_dir=tmp_path)


@pytest.mark.slow
@pytest.mark.dependency(depends=["test_landsat1_slow"], scope="session")
def test_mosaic_landsat_slow(process_landsat1_slow: list[Path], tmp_path: Path, brazil_clip: Path) -> None:
    """Test Landsat mosaic."""
    mosaic_path = tmp_path / "mosaic_landsat_slow"
    mosaic(
        process_landsat1_slow,
        output_dir=mosaic_path,
        clip=brazil_clip,
        keep_all_channels=True,
        fill_nodata=True,
        nodata_order=True,
    )
