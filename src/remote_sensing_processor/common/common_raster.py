"""Commonly used raster manipulation functions."""

from typing import Any, Optional, Union

import tarfile
import zipfile
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import xarray as xr

import geopandas as gpd
import odc.geo.xr  # noqa F401
import rasterio as rio
import rasterio.crs
import rioxarray as rxr
import shapely

from pystac import Item

from remote_sensing_processor.common.common_functions import persist
from remote_sensing_processor.common.types import crs_adapter


def load_dataset(dataset: Item, bands: Optional[list] = None, clip: Optional[Path] = None) -> xr.Dataset:
    """Read bands from STAC dataset."""
    if bands is None:
        bands = list(dataset.assets)

    files = []
    managers = []
    for band in bands:
        if band in dataset.assets:
            # files.append(rxr.open_rasterio(dataset.assets[band].href, chunks=True, lock=True, default_name=band))
            # raster = rxr.open_rasterio(dataset.assets[band].href, chunks=True, lock=True, band_as_variable=True)

            with rxr.open_rasterio(dataset.assets[band].href, chunks=True, lock=True) as raster:
                if hasattr(raster.rio, "_manager") and raster.rio._manager is not None:
                    managers.append(raster.rio._manager)

                nodata = raster.rio.nodata

                raster = raster.to_dataset("band")

                # Setting nodata values
                for b in raster:
                    raster[b].rio.write_nodata(nodata, inplace=True)

                if len(raster) > 1:
                    # Add long names if they not exist
                    for var in raster:
                        if "long_name" not in raster[var].attrs:
                            raster[var].attrs["long_name"] = band

                    # Rename duplicated long_names
                    count_dict = {}
                    for var in raster:
                        string = raster[var].attrs["long_name"]
                        if string in count_dict:
                            count_dict[string] += 1
                            raster[var].attrs["long_name"] = f"{string}_{count_dict[string]}"
                        else:
                            count_dict[string] = 0

                    # Rename variables in dataset
                    raster = raster.rename({i: f"{band}/{raster[i].attrs['long_name']}" for i in raster})

                else:
                    raster = raster.rename(dict.fromkeys(raster, band))

                files.append(raster)

    # Assert files have equal shapes
    assert_equal_shapes(files)

    img = xr.merge(files)

    bbox = get_initial_bbox(img, clip)
    img = clip_to_initial_bbox(img, bbox)
    img = persist(img.chunk("auto"))

    # Close files
    for file in files:
        file.close()
        # file.rio._manager.close()
    for manager in managers:
        manager.close()
    del files
    del managers
    return img


def prepare_nodata(
    raster: Union[xr.Dataset, xr.DataArray],
    nodata: Optional[Union[int, float]] = None,
    default_nodata: Optional[Union[int, float]] = None,
) -> tuple[Union[xr.Dataset, xr.DataArray], Union[int, float]]:
    """Read nodata from raster if nodata is None or set to a default value and then write nodata value to raster."""
    if isinstance(raster, xr.Dataset):
        # Getting nodata
        if nodata is None:
            try:
                for band in raster:
                    nodata = default_nodata if raster[band].rio.nodata is None else raster[band].rio.nodata
            except Exception:
                nodata = default_nodata
        # Writing nodata
        for band in raster:
            raster[band].rio.write_nodata(nodata, inplace=True)
    else:
        # Getting nodata
        if nodata is None:
            try:
                nodata = default_nodata if raster.rio.nodata is None else raster.rio.nodata
            except Exception:
                nodata = default_nodata
        # Writing nodata
        raster.rio.write_nodata(nodata, inplace=True)
    raster = persist(raster)
    return raster, nodata


def set_nodata_to_nan(raster: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """Replaces nodata values with NaN."""
    if isinstance(raster, xr.Dataset):
        for band in raster:
            nodata = raster[band].rio.nodata
            raster[band] = raster[band].where(raster[band] != nodata)
    else:
        nodata = raster.rio.nodata
        raster = raster.where(raster != nodata)
    return persist(raster)


def restore_nodata_from_nan(raster: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """Replaces NaN values with a value from rio.nodata.

    It also prevents values that are equal to rio.nodata from being treated as nodata by slightly increasing them.
    """
    if isinstance(raster, xr.Dataset):
        for band in raster:
            nodata = raster[band].rio.nodata
            # Setting nodata values to not nodata
            raster[band] = raster[band].where(raster[band] != nodata, raster[band] + 1e-5)
            # Restoring nodata values
            raster[band] = raster[band].fillna(nodata)
    else:
        nodata = raster.rio.nodata
        # Setting nodata values to not nodata
        raster = raster.where(raster != nodata, raster + 1e-5)
        # Restoring nodata values
        raster = raster.fillna(nodata)
    return persist(raster)


def clipf(
    raster: Union[xr.Dataset, xr.DataArray],
    clip: Path,
    **kwargs: Any,
) -> Union[xr.Dataset, xr.DataArray]:
    """Clips a raster by vector."""
    crs = raster.rio.crs
    shape = gpd.read_file(clip).to_crs(crs)
    shape = convert_3d_2d(shape)
    raster = raster.rio.clip(shape.geometry.values, **kwargs)
    return persist(raster)


def clean_reproject_name(raster: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """ODC reproject can add a weird name to a band, removing it."""
    if isinstance(raster, xr.Dataset):
        for band in raster:
            if raster[band].name is not None and "reproject" in raster[band].name:
                raster[band].name = None
    else:
        if raster.name is not None and "reproject" in raster.name:
            raster.name = None
    return raster


def reproject(
    raster: Union[xr.Dataset, xr.DataArray],
    crs: rasterio.crs.CRS,
    resample: Optional[str] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """Reprojects a raster to crs."""
    if resample is None:
        resample = "nearest"
    raster = raster.odc.reproject(crs, resampling=resample).chunk("auto")
    # ODC adds names to data arrays and sets spatial_ref to epsg. Removing it
    raster = clean_reproject_name(raster)
    raster.spatial_ref.values = np.asarray(0, "int32")
    # Explicitly set CRS
    raster = raster.rio.write_crs(crs)
    # raster = raster.rio.reproject(crs, resampling=resample)
    # raster = raster.chunk("auto")
    return persist(raster)


def reproject_match(
    raster: Union[xr.Dataset, xr.DataArray],
    reference_raster: Union[xr.Dataset, xr.DataArray],
    resample: Optional[str] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """Reproject a raster to match another raster."""
    if resample is None:
        resample = "nearest"
    raster = raster.odc.reproject(reference_raster.odc.geobox, resampling=resample).chunk("auto")
    # ODC adds names to data arrays and sets spatial_ref to epsg. Removing it
    raster = clean_reproject_name(raster)
    raster.spatial_ref.values = np.asarray(0, "int32")
    # Explicitly set CRS
    raster = raster.rio.write_crs(reference_raster.rio.crs)
    # raster = raster.rio.reproject_match(pan, resampling=resample)
    # raster = raster.chunk("auto")
    return persist(raster)


def write(raster: xr.DataArray, path: Path, compute: bool = True) -> None:
    """Writes a raster to GeoTiff."""
    return raster.rio.to_raster(
        path,
        compress="deflate",
        PREDICTOR=2,
        ZLEVEL=9,
        BIGTIFF="IF_SAFER",
        tiled=True,
        NUM_THREADS="ALL_CPUS",
        compute=compute,
        lock=True,
    )


def write_dataset(img: xr.Dataset, dataset: Item, json_path: Path) -> None:
    """Writes a dataset to one or multiple GeoTiffs."""
    ds_dict = {}
    for band in img:
        # Explicitly convert to string
        band = str(band)
        if "/" in band:
            if band.split("/")[0] in ds_dict:
                ds_dict[band.split("/")[0]].append(band)
            else:
                ds_dict[band.split("/")[0]] = [band]
        else:
            ds_dict[band] = band

    results = []
    for band in ds_dict:
        results.append(write(img[ds_dict[band]], json_path.parent / dataset.assets[band].href, compute=False))
    dask.compute(*results)


def get_initial_bbox(raster: Union[xr.Dataset, xr.DataArray], clip: Optional[Path] = None) -> Optional[list]:
    """To reduce the calculations, we load only the area we need. Firstly, we get a raster bbox."""
    if clip is not None:
        raster_crs = raster.rio.crs
        shape = gpd.read_file(clip).to_crs(raster_crs)
        shape = convert_3d_2d(shape)
        tb = shape.total_bounds
        # Adding a 10% offset to prevent data from being lost due to projection or resolution difference
        lon_offset = (tb[2] - tb[0]) * 0.1
        lat_offset = (tb[3] - tb[1]) * 0.1
        tb[0] = tb[0] - lon_offset
        tb[1] = tb[1] - lat_offset
        tb[2] = tb[2] + lon_offset
        tb[3] = tb[3] + lat_offset
        return [shapely.geometry.box(*tb)]
    return None


def clip_to_initial_bbox(
    raster: Union[xr.Dataset, xr.DataArray],
    bbox: Optional[list] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """To reduce the calculations, we load only the area we need. Secondly, we clip a raster to bbox."""
    if bbox is not None:
        return raster.rio.clip(bbox)
    return raster


def check_dtype(raster: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """Checks if dtype of every variable in the dataset is the same."""
    # Getting dtype
    if isinstance(raster, xr.Dataset):
        dtype = next(raster[b].dtype.name for b in raster)
    elif isinstance(raster, xr.DataArray):
        dtype = raster.dtype.name
    else:
        raise ValueError("Input is not a valid Xarray object.")
    # Because predictor = 2 works with float64 only when libtiff > 3.2.0 and default libtiff in ubuntu is 3.2.0
    if dtype == "float64":
        dtype = "float32"
    raster = raster.astype(dtype)
    return persist(raster)


def unpack_bitmap(img: xr.DataArray) -> xr.DataArray:
    """Unpack bitmap array."""
    array = img.data
    nof_bits = 8 * array.dtype.itemsize
    xshape = list(array.shape)
    array = array.reshape([-1, 1])
    msk = 2 ** np.arange(nof_bits, dtype=array.dtype).reshape([1, nof_bits])
    uint8_packed = (array & msk).astype(bool).astype(np.uint8)
    uint8_packed = uint8_packed.reshape([*xshape, nof_bits])
    uint8_packed = da.moveaxis(uint8_packed, -1, 0)
    img = img.expand_dims(dim={"bit": np.arange(nof_bits)}, axis=0)
    img.data = uint8_packed
    return img


def convert_3d_2d(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """This function converts 3d geometry to 2d geometry."""
    df.geometry = df.geometry.make_valid().force_2d()
    return df


def get_first_proj(path: Union[Path, Item]) -> rio.crs.CRS:
    """Get the CRS of the first file in a file list."""
    projection = None
    if isinstance(path, Item):
        if "proj:code" in path.properties:
            projection = crs_adapter.validate_python(path.properties["proj:code"])
        else:
            raise ValueError("No CRS in STAC dataset")
    else:
        img = None
        if path.is_dir():
            img = next(list(path.glob(e)) for e in ["**/*.tif", "*.TIF", "**/*.jp2"])
        elif (".tar" in path.suffixes) or (".gz" in path.suffixes):
            with tarfile.open(path) as file:
                for i in file.getmembers():
                    if any(s in (path / i.name).suffixes for s in [".tif", ".TIF", ".jp2"]):
                        img = "tar://" + (path / i.name).as_posix()
                        break
        elif ".zip" in path.suffixes:
            with zipfile.ZipFile(path) as file:
                bands = file.namelist()
                for band in bands:
                    if any(s in (path / band).suffixes for s in [".tif", ".TIF", ".jp2"]):
                        img = "zip://" + (path / band).as_posix()
                        break
        if img is None:
            raise ValueError("Cannot read CRS")
        with rio.open(img) as im:
            projection = im.crs
    if projection is None:
        raise ValueError("Cannot read CRS")
    return projection


def assert_equal_shapes(rasters: list[Union[xr.Dataset, xr.DataArray]]) -> None:
    """Assert if array shapes are equal."""
    shape = None
    for raster in rasters:
        if isinstance(raster, xr.Dataset):
            for band in raster:
                if shape is None:
                    shape = raster[band].shape[1:]
                elif raster[band].shape[1:] != shape:
                    raise ValueError(str(band) + " shape is not equal to other bands.")
        else:
            if shape is None:
                shape = raster.shape[1:]
            elif raster.shape[1:] != shape:
                raise ValueError(str(raster.name) + " shape is not equal to other bands.")


def make_nodata_equal(
    raster: Union[xr.Dataset, xr.DataArray],
    nodata: Optional[Union[int, float]] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """Remove data if there is no data at any band."""
    if nodata is None:
        if isinstance(raster, xr.Dataset):
            mask = xr.concat([raster[var].isnull() for var in raster.data_vars], dim="band").any(dim="band")
        else:
            mask = raster.isnull().any(dim="band")
        raster = raster.where(~mask)
    else:
        if isinstance(raster, xr.Dataset):
            mask = xr.concat([raster[var] == nodata for var in raster.data_vars], dim="band").any(dim="band")
        else:
            mask = (raster == nodata).any(dim="band")
        raster = raster.where(~mask, nodata)
    return raster
