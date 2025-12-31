"""Test vegetation index calculation."""

from pathlib import Path

import pytest

from pystac import Item

from remote_sensing_processor import calculate_index


@pytest.mark.dependency(depends=["test_landsat"], scope="session")
def test_ndvi(process_landsat: Path) -> None:
    """Test NDVI calculation."""
    calculate_index("NDVI", process_landsat)


@pytest.mark.dependency(depends=["test_landsat"], scope="session")
def test_evi2(process_landsat: Path) -> None:
    """Test EVI2 calculation."""
    calculate_index("EVI2", process_landsat, bands={"g": 2.5, "L": 1.25})


@pytest.mark.dependency(depends=["test_landsat"], scope="session")
def test_kndvi(process_landsat: Path) -> None:
    """Test kNDVI calculation."""
    calculate_index("kNDVI", process_landsat)


@pytest.mark.dependency(depends=["test_landsat"], scope="session")
def test_kevi(process_landsat: Path) -> None:
    """Test kEVI calculation."""
    calculate_index("kEVI", process_landsat, bands={"g": 2.5, "C1": 6.0, "C2": 7.5, "L": 0.5})


@pytest.mark.slow
@pytest.mark.dependency(depends=["test_landsat8_slow"], scope="session")
def test_ndvi_slow(process_landsat8_slow: Path) -> None:
    """Test NDVI calculation."""
    calculate_index("NDVI", process_landsat8_slow)


@pytest.mark.slow
@pytest.mark.dependency(depends=["test_landsat8_slow"], scope="session")
def test_evi2_slow(process_landsat8_slow: Path, load_landsat8_brazil: Item) -> None:
    """Test EVI2 calculation."""
    calculate_index("EVI2", process_landsat8_slow, bands={"g": 2.5, "L": 1.25})


@pytest.mark.slow
@pytest.mark.dependency(depends=["test_landsat8_slow"], scope="session")
def test_kndvi_slow(process_landsat8_slow: Path, load_landsat8_brazil: Item) -> None:
    """Test kNDVI calculation."""
    calculate_index("kNDVI", process_landsat8_slow)


@pytest.mark.slow
@pytest.mark.dependency(depends=["test_landsat8_slow"], scope="session")
def test_kevi_slow(process_landsat8_slow: Path, load_landsat8_brazil: Item) -> None:
    """Test kEVI calculation."""
    calculate_index("kEVI", process_landsat8_slow, bands={"g": 2.5, "C1": 6.0, "C2": 7.5, "L": 0.5})


@pytest.mark.slow
def test_custom_index_slow(tmp_path: Path, load_landsat_beijing: Item) -> None:
    """Test custom normalized difference index calculation."""
    item = load_landsat_beijing
    item.assets = {"nir": item.assets["nir08"], "red": item.assets["red"]}
    item.id = "lsdata"
    calculate_index("NDVI_Custom_remote", item, tmp_path, bands={"b1": "nir", "b2": "red"})
