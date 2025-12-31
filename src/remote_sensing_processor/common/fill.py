"""Raster gap filling algorithms."""

from typing import Optional, Union

import numpy as np
import xarray as xr

from rasterio.fill import fillnodata as rio_fillnodata

from remote_sensing_processor.common.common_functions import persist


def fillnodata(
    raster: Union[xr.Dataset, xr.DataArray],
    mask: Union[xr.Dataset, xr.DataArray],
    fill_distance: Optional[int] = 100,
    nodata: Optional[Union[int, float]] = None,
) -> xr.Dataset:
    """Fill the gaps in the raster."""
    raster = raster.where(mask != 0)
    raster = persist(raster)

    raster = xr.apply_ufunc(
        fillnodata_ufunc,
        raster,
        kwargs={"fill_distance": fill_distance, "nodata": nodata},
        dask="parallelized",
    )

    return persist(raster)


def fillnodata_ufunc(
    raster: np.ndarray,
    fill_distance: Optional[int] = 100,
    nodata: Optional[Union[int, float]] = None,
) -> xr.DataArray:
    """A wrapper around rasterio fillnodata function."""
    mask = ~np.isnan(raster)
    return rio_fillnodata(raster, mask=mask, max_search_distance=fill_distance, nodata=nodata)
