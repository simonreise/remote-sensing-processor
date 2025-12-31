"""Test replace values."""

from pathlib import Path

import pytest

import rioxarray as rxr

from pystac import Item

from remote_sensing_processor import rasterize


@pytest.mark.slow
def test_rasterize(tmp_path: Path, load_lulc_roma: Item, landmask: Path) -> None:
    """Test replacing single value."""
    rasterize_file = tmp_path / "rasterize.tif"
    item = load_lulc_roma
    rasterize(vector=landmask, reference_raster=item, value="is_land", output_path=rasterize_file, nodata=-1)
    with rxr.open_rasterio(rasterize_file) as raster:
        assert raster.rio.nodata == -1
        assert raster.max() == 1
        assert raster.min() == 0
