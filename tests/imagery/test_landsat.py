"""Test Landsat preprocessing."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor import landsat


@pytest.mark.dependency(name="test_landsat", scope="session")
def test_landsat(process_landsat: Path, landsat_file: Path) -> None:
    """Test processing Landsat."""
    name = Path(landsat_file).name.split(".")[0]
    assert process_landsat.name == name + ".json"


@pytest.mark.dependency(name="test_landsat8_slow", scope="session")
@pytest.mark.slow
def test_landsat8_slow(process_landsat8_slow: Path, load_landsat8_brazil: Item) -> None:
    """Test processing Landsat 8."""
    name = load_landsat8_brazil.id
    assert process_landsat8_slow.name == name + ".json"


@pytest.mark.slow
def test_landsat5_slow(tmp_path: Path, load_landsat5_brazil: Item, brazil_clip: Path) -> None:
    """Test processing Landsat 5."""
    item = load_landsat5_brazil
    out = landsat(item, output_path=tmp_path, clip=brazil_clip)
    name = item.id
    assert out[0] == tmp_path / name / (name + ".json")


@pytest.mark.dependency(name="test_landsat1_slow", scope="session")
@pytest.mark.slow
def test_landsat1_slow(process_landsat1_slow: list[Path], tmp_path: Path, load_landsat1_brazil: list[Item]) -> None:
    """Test processing Landsat 1."""
    name = load_landsat1_brazil[0].id
    assert process_landsat1_slow[0].name == name + ".json"
    name = load_landsat1_brazil[1].id
    assert process_landsat1_slow[1].name == name + ".json"
