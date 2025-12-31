"""Pytest configuration."""

pytest_plugins = [
    "tests.data_files.files",
    "tests.utils.load_dem",
    "tests.utils.load_landsat",
    "tests.utils.load_lulc",
    "tests.utils.load_sentinel2",
    "tests.imagery.landsat_fixtures",
]
