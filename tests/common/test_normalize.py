"""Test normalization."""

from pathlib import Path

import pytest

from numpy.testing import assert_allclose
from xarray.testing import assert_allclose as xr_assert_allclose

import rioxarray as rxr

from pystac import Item

from remote_sensing_processor import denormalize, get_normalization_params, normalize


def test_min_max_normalization_params(dem_toy_data: Path) -> None:
    """Test min-max normalization parameters retrieving."""
    minimum, maximum = get_normalization_params.min_max(dem_toy_data)
    assert_allclose(minimum, 3762.275634765625)
    assert_allclose(maximum, 5886.78662109375)


def test_z_score_normalization_params(dem_toy_data: Path) -> None:
    """Test z-score normalization parameters retrieving."""
    mean, stddev = get_normalization_params.z_score(dem_toy_data)
    assert_allclose(mean, 4918.69140625)
    assert_allclose(stddev, 487.67279052734375)


def test_dynamic_world_normalization_params(dem_toy_data: Path) -> None:
    """Test dynamic world normalization parameters retrieving."""
    norm_params = get_normalization_params.dynamic_world(dem_toy_data, percentiles=[30, 70])
    assert_allclose(norm_params[30], 8.435240840911865)
    assert_allclose(norm_params[70], 8.552703666687012)


def test_percentile_normalization_params(dem_toy_data: Path) -> None:
    """Test percentile normalization parameters retrieving."""
    norm_params = get_normalization_params.percentile(dem_toy_data, percentiles=[30, 70])
    assert_allclose(norm_params[30], 4606.57783203125)
    assert_allclose(norm_params[70], 5180.741748046875)


@pytest.mark.slow
def test_min_max_normalization_params_slow(load_dem_grand_canyon: Item) -> None:
    """Test min-max normalization parameters retrieving."""
    item = load_dem_grand_canyon
    minimum, maximum = get_normalization_params.min_max(item)
    assert_allclose(minimum, 528.0)
    assert_allclose(maximum, 2811.199951171875)


@pytest.mark.slow
def test_z_score_normalization_params_slow(load_dem_grand_canyon: Item) -> None:
    """Test z-score normalization parameters retrieving."""
    item = load_dem_grand_canyon
    mean, stddev = get_normalization_params.z_score(item)
    assert_allclose(mean, 1764.97998046875)
    assert_allclose(stddev, 419.1229553222656)


@pytest.mark.slow
def test_dynamic_world_normalization_params_slow(load_dem_grand_canyon: Item) -> None:
    """Test dynamic world normalization parameters retrieving."""
    item = load_dem_grand_canyon
    norm_params = get_normalization_params.dynamic_world(item, percentiles=[30, 70])
    assert_allclose(norm_params[30], 7.3412675857543945)
    assert_allclose(norm_params[70], 7.544810438156127)


@pytest.mark.slow
def test_percentile_normalization_params_slow(load_dem_grand_canyon: Item) -> None:
    """Test percentile normalization parameters retrieving."""
    item = load_dem_grand_canyon
    norm_params = get_normalization_params.percentile(item, percentiles=[30, 70])
    assert_allclose(norm_params[30], 1542.666015625)
    assert_allclose(norm_params[70], 1890.903991699218)


def test_min_max_normalize_denormalize(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test min-max normalization and denormalization."""
    norm_file = tmp_path / "norm_fast_min_max.tif"
    denorm_file = tmp_path / "denorm_fast_min_max.tif"
    normalize.min_max(dem_toy_data, output_path=norm_file, minimum=3000, maximum=6000)
    denormalize.min_max(norm_file, output_path=denorm_file, minimum=3000, maximum=6000)
    assert_files_equal(dem_toy_data, denorm_file)


def test_z_score_normalize_denormalize(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test z-score normalization and denormalization."""
    norm_file = tmp_path / "norm_fast_z_score.tif"
    denorm_file = tmp_path / "denorm_fast_z_score.tif"
    normalize.z_score(dem_toy_data, output_path=norm_file, mean=4919, stddev=488)
    denormalize.z_score(norm_file, output_path=denorm_file, mean=4919, stddev=488)
    assert_files_equal(dem_toy_data, denorm_file)


def test_dynamic_world_normalize_denormalize(tmp_path: Path, dem_toy_data: Path) -> None:
    """Test dynamic world normalization and denormalization."""
    norm_file = tmp_path / "norm_fast_dw.tif"
    denorm_file = tmp_path / "denorm_fast_dw.tif"
    normalize.dynamic_world(
        dem_toy_data,
        output_path=norm_file,
        percentile1=8.435240840911865,
        percentile2=8.552703666687012,
    )
    denormalize.dynamic_world(
        norm_file,
        output_path=denorm_file,
        percentile1=8.435240840911865,
        percentile2=8.552703666687012,
    )
    assert_files_equal(dem_toy_data, denorm_file)


@pytest.mark.slow
def test_min_max_normalize_denormalize_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test min-max normalization and denormalization."""
    norm_file = tmp_path / "norm_slow_min_max.tif"
    denorm_file = tmp_path / "denorm_toy_min_max.tif"
    item = load_dem_grand_canyon
    normalize.min_max(item, output_path=norm_file, minimum=0, maximum=3000)
    denormalize.min_max(norm_file, output_path=denorm_file, minimum=0, maximum=3000)
    assert_file_and_stac_equal(item, denorm_file)


@pytest.mark.slow
def test_z_score_normalize_denormalize_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test z-score normalization and denormalization."""
    norm_file = tmp_path / "norm_slow_z_score.tif"
    denorm_file = tmp_path / "denorm_slow_z_score.tif"
    item = load_dem_grand_canyon
    normalize.z_score(item, output_path=norm_file, mean=1765, stddev=420)
    denormalize.z_score(norm_file, output_path=denorm_file, mean=1765, stddev=420)
    assert_file_and_stac_equal(item, denorm_file)


@pytest.mark.slow
def test_dynamic_world_normalize_denormalize_slow(tmp_path: Path, load_dem_grand_canyon: Item) -> None:
    """Test dynamic world normalization and denormalization."""
    norm_file = tmp_path / "norm_slow_dw.tif"
    denorm_file = tmp_path / "denorm_slow_dw.tif"
    item = load_dem_grand_canyon
    normalize.dynamic_world(item, output_path=norm_file, percentile1=7.3412675857543945, percentile2=7.544810438156127)
    denormalize.dynamic_world(
        norm_file,
        output_path=denorm_file,
        percentile1=7.3412675857543945,
        percentile2=7.544810438156127,
    )
    assert_file_and_stac_equal(item, denorm_file)


def assert_files_equal(file1: Path, file2: Path) -> None:
    """Assert if two files are almost equal."""
    with rxr.open_rasterio(file1) as raster1, rxr.open_rasterio(file2) as raster2:
        xr_assert_allclose(raster1, raster2)


def assert_file_and_stac_equal(stac: Item, file2: Path) -> None:
    """Assert if two files are almost equal."""
    file1 = stac.assets[next(iter(stac.assets))].href
    with rxr.open_rasterio(file1) as raster1, rxr.open_rasterio(file2) as raster2:
        xr_assert_allclose(raster1, raster2)
