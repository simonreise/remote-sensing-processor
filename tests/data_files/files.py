"""Fixtures for files."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Data directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def dem_toy_data(data_dir: Path) -> Path:
    """DEM toy data."""
    return data_dir / "DEM_toy_data.tif"


@pytest.fixture(scope="session")
def esa_worldcover_toy_data(data_dir: Path) -> Path:
    """ESA WorldCover toy data."""
    return data_dir / "ESA_WorldCover_toy_data.tif"


@pytest.fixture(scope="session")
def esa_worldcover_toy_data_clip(data_dir: Path) -> Path:
    """ESA WorldCover toy data clip."""
    return data_dir / "ESA_WorldCover_toy_data_clip.gpkg"


@pytest.fixture(scope="session")
def esa_worldcover_toy_data_clip1(data_dir: Path) -> Path:
    """ESA WorldCover toy data clip 1."""
    return data_dir / "ESA_WorldCover_toy_data_clip1.gpkg"


@pytest.fixture(scope="session")
def esa_worldcover_toy_data_clip2(data_dir: Path) -> Path:
    """ESA WorldCover toy data clip 2."""
    return data_dir / "ESA_WorldCover_toy_data_clip2.gpkg"


@pytest.fixture(scope="session")
def beijing_clip(data_dir: Path) -> Path:
    """Beijing clip."""
    return data_dir / "beijing.gpkg"


@pytest.fixture(scope="session")
def brazil_clip(data_dir: Path) -> Path:
    """Brazil clip."""
    return data_dir / "brazil.gpkg"


@pytest.fixture(scope="session")
def india_clip(data_dir: Path) -> Path:
    """India clip."""
    return data_dir / "india.gpkg"


@pytest.fixture(scope="session")
def roma_clip(data_dir: Path) -> Path:
    """Roma clip."""
    return data_dir / "roma.gpkg"


@pytest.fixture(scope="session")
def landmask(data_dir: Path) -> Path:
    """Landmask."""
    return data_dir / "landmask.gpkg"


@pytest.fixture(scope="session")
def landsat_file(data_dir: Path) -> Path:
    """Landsat file."""
    return data_dir / "LC08_L2SP_005009_20150710_20200908_02_T2.tar"
