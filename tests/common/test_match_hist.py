"""Test matching histograms."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor import match_hist


@pytest.mark.slow
def test_match_hist_slow(tmp_path: Path, load_landsat_beijing_match_hist: tuple[Item, Item]) -> None:
    """Test matching histograms of two Landsat images."""
    match_hist_file = tmp_path / "match_hist_slow.tif"
    item, item1 = load_landsat_beijing_match_hist
    match_hist(item, reference_raster=item1, output_path=match_hist_file)
