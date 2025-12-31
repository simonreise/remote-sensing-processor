"""Test Sentinel2 preprocessing."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor import sentinel2


@pytest.mark.slow
def test_sentinel2_slow(tmp_path: Path, load_sentinel2_india: Item, india_clip: Path) -> None:
    """Test processing Sentinel2."""
    item = load_sentinel2_india
    sentinel2(item, output_path=tmp_path, clip=india_clip, upscale="resample")
