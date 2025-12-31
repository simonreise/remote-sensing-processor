"""General tiles generation functions."""

from pydantic import BaseModel
from typing import Optional, Union

import warnings
from pathlib import Path

from xbatcher import BatchGenerator

import dask
import numpy as np
import xarray as xr

from pystac import Item

from datasets import Dataset

from remote_sensing_processor.common.common_functions import create_folder, delete, persist
from remote_sensing_processor.common.common_raster import (
    assert_equal_shapes,
    load_dataset,
    make_nodata_equal,
    prepare_nodata,
    reproject_match,
    write,
)
from remote_sensing_processor.common.dataset import read_dataset
from remote_sensing_processor.common.rasterize import rasterize_func


def create_folders(output: Path, split: dict[str, Union[int, float]]) -> None:
    """Create basic folder structure for rsp dataset."""
    create_folder(output)
    for name in split:
        create_folder(output / name)


def write_reference(img: xr.DataArray, output: Path, nodata: Optional[Union[int, float]]) -> None:
    """Write a reference file to make it possible to map predictions."""
    ref = xr.zeros_like(img[0], dtype="uint8")
    ref = ref.where(img[0] == nodata, 1)
    ref, _ = prepare_nodata(ref, 0)
    write(ref, output / "ref.tif")


def prepare_images(
    img: list[Union[Path, Item]],
    nodata: Optional[Union[int, float]],
    dtype: Optional[np.dtype],
) -> tuple[xr.DataArray, Optional[Union[int, float]]]:
    """Prepare image data."""
    stacs = []
    for path in img:
        stacs.append(read_dataset(path))
    datasets = []
    for stac in stacs:
        datasets.append(dask.delayed(load_dataset)(stac))
    datasets = list(dask.compute(*datasets))
    assert_equal_shapes(datasets)
    datasets = add_prefixes(datasets)
    img = xr.merge(datasets)
    # Fixing x_nodata if different nodata values
    if nodata is None:
        nodata = get_nodata(img)
    img, nodata = fix_nodata(img, nodata)
    img = make_nodata_equal(img, nodata)
    img = check_dtype(img=img, dtype=dtype)
    img = persist(img.squeeze().to_array("band").chunk("auto"))
    return img, nodata


def prepare_vector_sm(
    path: Path,
    ref: xr.DataArray,
    burn_value: str,
    name: str,
    y_nodata: Optional[Union[int, float]],
) -> xr.DataArray:
    """Prepare vector segmentation map."""
    arr = rasterize_func(path, ref, burn_value, y_nodata)
    arr = arr[burn_value]
    arr.name = name
    return arr


def prepare_raster_sm(
    path: Union[Path, Item],
    ref: xr.DataArray,
    name: str,
    y_nodata: Optional[Union[int, float]],
) -> xr.Dataset:
    """Prepare raster segmentation map."""
    stac = read_dataset(path)
    ds = load_dataset(stac)
    if ds.shape[0] != 1:
        raise ValueError("Looks like you are trying to use multiband raster as a target variable")
    ds, y_nodata = fix_nodata(ds, y_nodata)
    ds = reproject_match(ds, ref)
    ds.name = name
    return ds


def add_prefixes(datasets: list[xr.Dataset]) -> list[xr.Dataset]:
    """Add dataset name as a prefix to prevent errors if there are same bands in different datasets."""
    for i in range(len(datasets)):
        datasets[i] = datasets[i].rename(
            {j: str(i) + "_" + str(j) for j in datasets[i].data_vars},
        )
    return datasets


def check_dtype(img: xr.Dataset, dtype: np.dtype = None, dtype_class: np.dtype = None) -> xr.Dataset:
    """Check if every band in the dataset is of specific dtype."""
    if dtype is None and dtype_class is not None and not all(np.issubdtype(img[b].dtype, dtype_class) for b in img):
        if dtype_class is np.integer:
            dtype = "int32"
        elif dtype_class is np.floating:
            dtype = "float32"
    for band in img:
        if dtype is not None:
            img[band] = img[band].astype(dtype)
    return img


def border_pad(img: xr.DataArray, tile_size: int) -> tuple[int, list[int]]:
    """Get the padding size for pad function."""
    # Calculate border
    border = round(tile_size * 0.0625)

    # Calculate padding
    shp_in = img.shape[1:3]
    shp_pad = []
    for d in shp_in:
        shp_pad.append(tile_size - (border * 2) - (d % (tile_size - (border * 2))))
    return border, shp_pad


def pad(img: xr.DataArray, shp_pad: list[int], nodata: Optional[Union[int, float]] = None) -> xr.DataArray:
    """Pad array to make it big enough to load the last batch."""
    img = img.pad({"y": (0, shp_pad[0]), "x": (0, shp_pad[1])}, mode="constant", constant_values=nodata)
    return persist(img)


def filter_samples(batcher: BatchGenerator, samples: list[int], nodata: Optional[Union[int, float]]) -> list[int]:
    """Remove samples that contain only nodata."""

    @dask.delayed
    def is_not_empty_batch(data: xr.DataArray) -> bool:
        return (data != nodata).any().item()

    tasks = []
    for i in samples:
        batch = batcher[i]  # may already be a Dask array or NumPy array
        tasks.append(is_not_empty_batch(batch))
    results = dask.compute(*tasks)
    return [i for i, keep in zip(samples, results, strict=True) if keep]


def split_samples(samples: list[int], split: dict[str, Union[int, float]]) -> dict[str, list[int]]:
    """Split dataset to subdatasets."""
    out_samples = {}
    j = 0
    for k, v in split.items():
        out_samples[k] = samples[j : j + round(v / sum(split.values()) * len(samples))]
        j = j + round(v / sum(split.values()) * len(samples))
    return out_samples


def normalize_classes(img: xr.DataArray, nodata: Union[int, float], classes: list[int]) -> xr.DataArray:
    """Replacing classes values to 0, 1, 2, ..., n. Now is not used."""
    # TODO: Remove it, because it is not used anymore
    nodata = classes.index(nodata)
    if {k == v for k, v in enumerate(classes)} != {True}:
        for v, k in enumerate(classes):
            img = xr.where(img == k, v + 100000, img)
        for v, _ in enumerate(classes):
            img = xr.where(img == v + 100000, img - 100000, img)
    img = img.rio.write_nodata(nodata)
    return persist(img)


def get_nodata(x: xr.Dataset) -> Optional[Union[int, float]]:
    """Gets the most common nodata value."""
    nodatas = []
    for band in x:
        if x[band].rio.nodata is not None:
            nodatas.append(x[band].rio.nodata)
    return None if len(nodatas) == 0 else max(set(nodatas), key=nodatas.count)


def fix_nodata(img: xr.Dataset, nodata: Optional[Union[int, float]]) -> tuple[xr.Dataset, Union[int, float]]:
    """Overwrite nodata if nodata value is different from default nodata.."""
    for band in img:
        if img[band].rio.nodata != nodata:
            warnings.warn(
                str(band)
                + " nodata value is "
                + str(img[band].rio.nodata)
                + ". It will be converted to "
                + str(nodata),
                stacklevel=2,
            )
            img[band] = img[band].where(img[band] != nodata, img[band] + 1e-5)
            img[band] = img[band].where(img[band] != img[band].rio.nodata, nodata)
    img, nodata = prepare_nodata(img, nodata, 0)
    return img, nodata


def check_classes(rasters: xr.Dataset, nodata: Optional[Union[int, float]]) -> None:
    """Check the classes format."""
    if nodata != 0:
        warnings.warn(
            "Looks like y_nodata is not 0. The recommended data format is (0, 1, 2, 3, ..., n).",
            stacklevel=2,
        )
    for raster in rasters:
        classes = sorted(np.unique(rasters[raster]))
        for i in range(len(classes) - 1):
            if classes[i + 1] - classes[i] > 2:
                warnings.warn(
                    "Looks like classes in"
                    + str(rasters[raster].name)
                    + " are sparse. Training could be slow. "
                    + "If classes sparsity is not for purpose, please change class values to recommended format "
                    + " (0, 1, 2, 3, ..., n).",
                    stacklevel=2,
                )


def replace_y_in_meta(meta: dict, dataset: BaseModel) -> dict:
    """Replaces dict of different y values to a single y."""
    # If y is not needed (e.g., if predict)
    if dataset.predict:
        del meta["y"]
        return meta

    if len(meta["y"]) > 1:
        if dataset.y is None:
            raise ValueError("y not found in dataset")
    elif dataset.y is None:
        dataset.y = next(iter(meta["y"].keys()))

    meta["y"] = meta["y"][dataset.y]
    meta["y"]["name"] = dataset.y
    return meta


def get_cache(ds: Dataset) -> list[Path]:
    """Get temp .cache folder path."""
    ds.cleanup_cache_files()

    # Get the cache file paths
    cache_files = ds.cache_files

    files = []
    # Delete each cache file and its associated directory
    for cache_file in cache_files:
        cache_dir = Path(cache_file["filename"])
        for i in range(len(cache_dir.parents)):
            if cache_dir.parents[i].name == ".cache":
                files.append(cache_dir.parents[i])
    return files


def clean_cache(files: list[Path]) -> None:
    """Delete temp .cache folder."""
    for file in files:
        delete(file)
