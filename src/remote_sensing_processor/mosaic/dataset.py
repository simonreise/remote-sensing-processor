"""Mosaic STAC dataset functions."""

import datetime
from pathlib import Path

import pyproj

from xarray import Dataset

from pystac import Item
from stactools.core.utils import antimeridian

from remote_sensing_processor.common.dataset import filter_bands, validate


def postprocess_mosaic_dataset(
    datasets: list[Item],
    final: Dataset,
    output_dir: Path,
    bands: list[str],
) -> tuple[Item, Path]:
    """Postprocess mosaic dataset."""
    # Initializing dataset
    stac = datasets[0]
    stac.id = stac.id + "_mosaic"
    stac.datetime = datetime.datetime.now()

    # Deleting unneeded assets
    filter_bands(stac, bands)

    # Adding bands that were not present in the first dataset
    for band in bands:
        if band not in stac.assets:
            for ds in datasets:
                if band in ds.assets:
                    stac.assets[band] = ds.assets[band]

    # Adding hrefs
    for band in bands:
        stac.assets[band].href = stac.assets[band].ext.eo.bands[0].name + ".tif"

    # Changing datatypes and nodata
    for band in bands:
        if stac.assets[band].ext.has("raster") and stac.assets[band].ext.raster.bands is not None:
            if stac.assets[band].ext.raster.bands[0].data_type is not None:
                stac.assets[band].ext.raster.bands[0].data_type = final[band].dtype.name
            if stac.assets[band].ext.raster.bands[0].nodata is not None and final[band].rio.nodata is not None:
                stac.assets[band].ext.raster.bands[0].nodata = float(final[band].rio.nodata)

    # Updating projection info
    stac.ext.proj.shape = list(final[bands[0]].shape[-2:])
    stac.ext.proj.transform = list(final[bands[0]].rio.transform())[:6]
    stac.ext.proj.epsg = final[bands[0]].rio.crs.to_epsg()
    transformer = pyproj.Transformer.from_crs(final[bands[0]].rio.crs, "EPSG:4326")
    stac.bbox = list(transformer.transform_bounds(*final[bands[0]].rio.bounds()))
    stac.geometry["coordinates"] = (
        (
            (stac.bbox[0], stac.bbox[1]),  # LL
            (stac.bbox[2], stac.bbox[1]),  # LR
            (stac.bbox[2], stac.bbox[3]),  # UR
            (stac.bbox[2], stac.bbox[3]),  # UL
            (stac.bbox[0], stac.bbox[1]),  # LL
        ),
    )

    # Updating projection info for each asset
    for band in bands:
        # Changing projection info if resample is not None
        if stac.assets[band].ext.proj.shape is not None:
            stac.assets[band].ext.proj.shape = list(final[band].shape[-2:])
        if stac.assets[band].ext.proj.transform is not None:
            stac.assets[band].ext.proj.transform = list(final[band].rio.transform())[:6]

    # Adding self link
    json_path = output_dir / (stac.id + ".json")
    stac.clear_links()
    stac.set_self_href(json_path.as_posix())

    # Fix geometries if needed
    stac = antimeridian.fix_item(stac, antimeridian.Strategy.SPLIT)
    # Validation
    validate(stac)
    return stac, json_path
