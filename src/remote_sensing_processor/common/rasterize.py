"""Rasterize a vector file."""

from pydantic import validate_call
from typing import Optional, Union

from pathlib import Path

import numpy as np
from xarray import DataArray

import geopandas as gpd
import rioxarray as rxr
from geocube.api.core import make_geocube

from remote_sensing_processor.common.common_functions import create_path, persist
from remote_sensing_processor.common.common_raster import (
    check_dtype,
    convert_3d_2d,
    write,
)
from remote_sensing_processor.common.dataset import add_asset, add_rsp_metadata, postprocess_dataset, read_dataset
from remote_sensing_processor.common.types import DirectoryPath, FilePath, NewPath, PystacItem


@validate_call
def rasterize(
    vector: FilePath,
    reference_raster: Union[FilePath, DirectoryPath, PystacItem],
    value: str,
    output_path: Union[FilePath, DirectoryPath, NewPath],
    nodata: Optional[Union[int, float]] = 0,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Rasterizes a vector file.

    Parameters
    ----------
    vector : path to vector as a string
        Path to a vector file that needs to be rasterized.
    reference_raster : string or STAC Item
        Path to a raster file to get shape, resolution and projection from.
    value : string
        A field to use for a burn-in value. Field should be numeric.
    output_path : string
        Path to an output file.
    nodata: int or float (default = 0)
        A value that will be used as nodata.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.rasterize(
        ...     vector="/home/rsp_test/mosaics/treecover/treecover.shp",
        ...     reference_raster="/home/rsp_test/mosaics/sentinel/B1.tif",
        ...     value="tree_species",
        ...     output_path="/home/rsp_test/mosaics/treecover/treecover.tif",
        ...     nodata=0,
        ... )
    """
    raster = read_dataset(reference_raster)
    rr = raster.assets[next(iter(raster.assets.keys()))].href
    with rxr.open_rasterio(rr, chunks=True, lock=True) as tif:
        # Read raster
        rasterized = rasterize_func(vector, tif, value, nodata)
        rasterized = check_dtype(rasterized)

    # Creating an output folder
    create_path(output_path)

    # Adding a band with the name of a burn value
    add_asset(
        item=raster,
        name=value,
        path=value + ".tif",
    )
    add_rsp_metadata(raster, rsp_type="Undefined")
    # Creating final STAC dataset
    dataset, json_path = postprocess_dataset(
        raster,
        rasterized,
        output_path,
        bands=[value],
    )

    # Write
    write(rasterized[value], json_path.parent / dataset.assets[value].href)

    if write_stac:
        # Writing JSON metadata file
        dataset.save_object(dest_href=json_path.as_posix())
        return json_path
    return output_path


def rasterize_func(
    vector: Path,
    raster: DataArray,
    burn_value: str,
    nodata: Optional[Union[int, float]] = 0,
) -> DataArray:
    """Rasterize function itself."""
    vector = gpd.read_file(vector)
    vector = convert_3d_2d(vector)
    if not np.issubdtype(vector[burn_value].dtype, np.number):
        raise TypeError("Burn value is not of a numeric type")
    rasterized = make_geocube(
        vector,
        measurements=[burn_value],
        like=raster,
        fill=nodata,
    )
    rasterized = persist(rasterized)
    rasterized = rasterized.chunk("auto")
    return persist(rasterized)
