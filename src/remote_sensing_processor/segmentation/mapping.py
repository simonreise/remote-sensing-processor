"""Basic mapping functions."""

from typing import Union

import warnings
from pathlib import Path

import xarray as xr

import rioxarray as rxr

from pystac import Item

from remote_sensing_processor.common.common_functions import persist
from remote_sensing_processor.common.dataset import add_asset, add_rsp_metadata, postprocess_dataset, read_dataset
from remote_sensing_processor.segmentation.segmentation import DataModule


def load_reference(dm: DataModule) -> tuple[xr.DataArray, Item]:
    """Load reference raster."""
    dataset = read_dataset(dm.reference)
    rr = dataset.assets[next(iter(dataset.assets.keys()))].href
    with rxr.open_rasterio(rr, chunks=True, lock=True) as tif:
        reference = persist(tif)
        try:
            reference = reference.compute()
        except Exception:
            warnings.warn("Can not load reference array to memory. Computation could be much slower.", stacklevel=1)
    return reference, dataset


def restore_classes(
    raster: xr.DataArray,
    classes: list[int],
    nodata: Union[int, float],
) -> tuple[xr.DataArray, Union[int, float]]:
    """Restore classes from normalized. Now is not used."""
    # TODO: No longer used. Can be removed.
    if classes is not None and {k == v for k, v in enumerate(classes)} != {True}:
        for k, v in enumerate(classes):
            raster = xr.where(raster == k, v + 100000, raster)
        for _, v in enumerate(classes):
            raster = xr.where(raster == v + 100000, raster - 100000, raster)
        # Convert nodata back to value from index
        nodata = classes[nodata]
    return raster, nodata


def post_process_raster_dataset(dataset: Item, raster: xr.DataArray, output: Path) -> tuple[Item, Path]:
    """Post-process STAC of a generated map."""
    raster = raster.to_dataset("band").expand_dims("band").rename_vars({1: output.stem})
    # Adding a band with the name of a burn value
    add_asset(
        item=dataset,
        name=output.stem,
        path=output.name,
    )
    add_rsp_metadata(dataset, rsp_type="Undefined")
    # Creating final STAC dataset
    dataset, json_path = postprocess_dataset(
        dataset,
        raster,
        output,
        bands=[output.stem],
    )
    return dataset, json_path
