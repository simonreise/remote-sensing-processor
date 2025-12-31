"""Fixtures for testing Landsat."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor import landsat


@pytest.fixture(scope="session")
def process_landsat(tmp_path_factory: pytest.TempPathFactory, landsat_file: Path) -> Path:
    """Process Landsat."""
    out_dir = tmp_path_factory.mktemp("landsat")
    out = landsat(landsat_file, output_path=out_dir)
    return out[0]


@pytest.fixture(scope="session")
def process_landsat8_slow(
    tmp_path_factory: pytest.TempPathFactory,
    load_landsat8_brazil: Item,
    brazil_clip: Path,
) -> Path:
    """Process Landsat 8."""
    out_dir = tmp_path_factory.mktemp("landsat8")
    out = landsat(load_landsat8_brazil, output_path=out_dir, clip=brazil_clip)
    return out[0]


@pytest.fixture(scope="session")
def process_landsat1_slow(
    tmp_path_factory: pytest.TempPathFactory,
    load_landsat1_brazil: list[Item],
    brazil_clip: Path,
) -> list[Path]:
    """Process Landsat 1."""
    out_dir = tmp_path_factory.mktemp("landsat1")
    return landsat(load_landsat1_brazil, output_path=out_dir, clip=brazil_clip)
