"""Mosaic rasters."""

from pydantic import PositiveInt, validate_call
from typing import Optional, Union

import dask
import dask.array as da
import xarray as xr

import rasterio as rio
import rioxarray as rxr
from rioxarray.merge import merge_datasets

from pystac import Item

from remote_sensing_processor.common.common_functions import create_folder, persist
from remote_sensing_processor.common.common_raster import (
    check_dtype,
    clipf,
    load_dataset,
    prepare_nodata,
    reproject,
    reproject_match,
    write_dataset,
)
from remote_sensing_processor.common.dataset import is_multiband, read_dataset
from remote_sensing_processor.common.fill import fillnodata
from remote_sensing_processor.common.match_hist import histogram_match
from remote_sensing_processor.common.types import CRS, DirectoryPath, FilePath, NewPath, PystacItem
from remote_sensing_processor.mosaic.dataset import postprocess_mosaic_dataset


@validate_call
def mosaic(
    inputs: list[Union[FilePath, DirectoryPath, PystacItem]],
    output_dir: Union[DirectoryPath, NewPath],
    fill_nodata: Optional[bool] = False,
    fill_distance: Optional[PositiveInt] = 250,
    clip: Optional[FilePath] = None,
    crs: Optional[CRS] = None,
    nodata: Optional[Union[int, float]] = None,
    reference_raster: Optional[Union[FilePath, DirectoryPath, PystacItem]] = None,
    resample: Optional[str] = "average",
    nodata_order: Optional[bool] = False,
    match_hist: Optional[bool] = False,
    keep_all_channels: Optional[bool] = True,
    write_stac: Optional[bool] = True,
) -> Union[DirectoryPath, NewPath]:
    """
    Creates mosaic from several rasters.

    Parameters
    ----------
    inputs : list of strings or list of STAC Items
        List of pathes to rasters to be merged or to folders where multiband imagery data is stored or to STAC Items
        in order from images that should be on top to images that should be on bottom.
    output_dir: path to output directory as a string
        Path where mosaic raster or rasters will be saved.
    fill_nodata : bool (default = False)
        Is filling the gaps in the raster needed.
    fill_distance : int (default = 250)
        Fill distance for `fill_nodata` function.
    clip : string (optional)
        Path to a vector file to be used to crop the image.
    crs : string (optional)
        CRS in which output data should be.
    nodata : int or float (default = None)
        Nodata value. If not set, then is read from inputs.
    reference_raster : string or STAC Item (optional)
        Reference raster is needed to bring output mosaic raster to the same resolution and projection
        as another data source.
        It is useful when you need to use data from different sources together.
    resample : resampling method from rasterio as a string (default = 'average')
        Resampling method that will be used to reproject and reshape to a reference raster shape.
        You can read more about resampling methods
        `here <https://rasterio.readthedocs.io/en/latest/topics/resampling.html>`_.
        Use 'nearest' if you want to keep only the same values that exist in the input raster.
    nodata_order : bool (default = False)
        Is it needed to merge images in order from images with less nodata on top (they are usually clear)
        to images with more nodata values on bottom (they are usually the most distorted and cloudy).
    match_hist : bool (default = False)
        Is it needed to match histograms of merged images. Improve mosaic uniformity, but change the original data.
    keep_all_channels : bool (default = True)
        Is needed only when you are merging images that have different number of channels
        (e.g., Landsat images from different generations).
        If True, all bands are processed, if False, only bands that are present in all input images are processed
        and others are omitted.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path to mosaic rasters.

    Examples
    --------
        >>> # Mosaic multiple Sentinel 2 multi-band products
        >>> import remote_sensing_processor as rsp
        >>> input_sentinels = [
        ...     "/home/rsp_test/sentinels/Sentinel1/meta.json",
        ...     "/home/rsp_test/sentinels/Sentinel2/meta.json",
        ...     "/home/rsp_test/sentinels/Sentinel3/meta.json",
        ...     "/home/rsp_test/sentinels/Sentinel4/meta.json",
        ...     "/home/rsp_test/sentinels/Sentinel5/meta.json",
        ...     "/home/rsp_test/sentinels/Sentinel6/meta.json",
        ... ]
        >>> border = "/home/rsp_test/border.gpkg"
        >>> mosaic_sentinel = rsp.mosaic(
        ...     inputs=input_sentinels,
        ...     output_dir="/home/rsp_test/mosaics/sentinel/",
        ...     clip=border,
        ...     crs="EPSG:4326",
        ...     nodata_order=True,
        ... )
        Processing completed
        >>> print(mosaic_sentinel)
        '/home/rsp_test/mosaics/sentinel/S2A_MSIL2A_20210821T064626_N0209_R063_T42VWR_20210821T064626_mosaic.json'

        >>> from glob import glob
        >>> # Mosaic multiple DEM files and matching it with a reference raster (Sentinel band)
        >>> lcs = glob("/home/rsp_test/landcover/*.tif")
        >>> print(lcs)
        ['/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N60E075_Map.tif',
         '/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N63E072_Map.tif',
         '/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N63E075_Map.tif']
        >>> mosaic_landcover = rsp.mosaic(
        ...     inputs=lcs,
        ...     output_dir="/home/rsp_test/mosaics/landcover/",
        ...     clip=border,
        ...     reference_raster="/home/rsp_test/mosaics/sentinel/B1.tif",
        ...     nodata=-1,
        ... )
        Processing completed
        >>> print(mosaic_landcover)
        '/home/rsp_test/mosaics/landcover/ESA_WorldCover_10m_2020_v100_N60E075_Map_mosaic.json'
    """
    # Reading datasets
    datasets = [read_dataset(i) for i in inputs]
    mb = is_multiband(datasets[0])

    # If datasets are single-band, then we should give the same name to their assets
    if not mb:
        bname = next(iter(datasets[0].assets.keys())) + "_mosaic"
        for i in range(len(datasets)):
            datasets[i].assets[bname] = datasets[i].assets.pop(next(iter(datasets[i].assets.keys())))
            datasets[i].assets[bname].ext.eo.bands[0].name = bname

    # Getting nodata value
    if nodata is None:
        nodata = get_nodata(datasets, nodata)

    # Sorting in nodata order
    if nodata_order:
        datasets = order(datasets, nodata, clip)

    # Reading reference raster dataset
    if reference_raster is not None:
        reference_raster = read_dataset(reference_raster)
        crs = rio.crs.CRS.from_user_input(reference_raster.ext.proj.code)

    # Getting only the bands we need
    bands = get_bands(datasets, keep_all_channels)

    # Creating an output folder if not exists
    create_folder(output_dir, clean=False)

    # Step 1. Pre-process the data
    futures = []
    for dataset in datasets:
        futures.append(
            dask.delayed(initial_process_dataset)(
                dataset=dataset,
                bands=bands,
                clip=clip,
                crs=crs,
                nodata=nodata,
            ),
        )
    files = list(dask.compute(*futures))

    # Step 2: If histogram matching is needed, apply it
    if match_hist:
        hist_match_futures = []
        # Apply histogram matching to the rest of the files in parallel
        for file in files[1:]:
            hist_match_futures.append(histogram_match(file, files[0], nodata))

        files[1:] = list(dask.compute(*hist_match_futures))

    # Merging files
    final = merge_datasets(files, method="first", nodata=nodata).chunk("auto")
    final = persist(final)

    # Resampling to the same shape and resolution as another raster
    if reference_raster is not None:
        ref = load_dataset(reference_raster)
        final = reproject_match(final, ref, resample)

    # Clipping mosaic with vector mask
    if clip is not None:
        final = clipf(final, clip)

    # Filling nodata
    if fill_nodata:
        mask = xr.ones_like(final)
        # nodata -> 0, data -> 1
        mask = mask.where(final != nodata, 0)
        # outside -> 1
        for band in mask:
            mask[band] = mask[band].rio.write_nodata(1)
        mask = clipf(mask, clip)
        # Fill nodata
        final = fillnodata(final, mask, fill_distance, nodata)

    final = check_dtype(final)

    # Creating final STAC dataset
    stac, json_path = postprocess_mosaic_dataset(datasets, final, output_dir, bands)

    # Write
    write_dataset(final, stac, json_path)

    if write_stac:
        # Writing JSON metadata file
        stac.save_object(dest_href=json_path.as_posix())
        return json_path
    return output_dir


def initial_process_dataset(
    dataset: Item,
    bands: list[str],
    clip: Optional[FilePath] = None,
    crs: Optional[CRS] = None,
    nodata: Optional[Union[int, float]] = None,
) -> xr.Dataset:
    """Prepare a single dataset for mosaicking, without histogram matching."""
    img = load_dataset(dataset, bands, clip)

    img, nodata = prepare_nodata(img, nodata=nodata)

    if crs is not None:
        img = reproject(img, crs)
    if clip is not None:
        img = clipf(img, clip)
    img = check_dtype(img)

    # Adding empty data arrays for bands that are absent in the current dataset
    for band in bands:
        if band not in dataset.assets:
            img[band] = img[next(iter(img.keys()))]
            img[band].data = da.full_like(img[band], nodata)
    return persist(img)


def get_bands(datasets: list[Item], keep_all_channels: bool) -> list[str]:
    """Read band names.

    If keep_all_channels == True then will read all the names,
    if False then will read only the bands that are present in every dataset.
    """
    band_lists = []
    for dataset in datasets:
        band_lists.append(list(dataset.assets.keys()))
    if keep_all_channels:
        final_bands = list({x for xs in band_lists for x in xs})
    else:
        final_bands = []
        for lst in band_lists:
            for i in lst:
                n = 0
                for ll in band_lists:
                    if i in ll:
                        n += 1
                if n == len(band_lists):
                    final_bands.append(i)
    return final_bands


def get_nodata(datasets: list[Item], nodata: Optional[Union[int, float]] = None) -> Union[int, float]:
    """Get nodata from multiple datasets."""
    for ds in datasets:
        bands = list(ds.assets.keys())
        for band in bands:
            with rxr.open_rasterio(ds.assets[band].href, chunks=True, lock=True) as img:
                if nodata is None:
                    nodata = img.rio.nodata
                else:
                    if nodata != img.rio.nodata:
                        raise ValueError("Nodata value of " + ds.id + " is different from the other files.")
    return nodata


def order(
    datasets: list[Item],
    nodata: Optional[Union[int, float]] = None,
    clip: Optional[FilePath] = None,
) -> list[Item]:
    """Sort datasets in order from least nodata to most nodata."""
    zeros = []
    for ds in datasets:
        band = next(iter(ds.assets.keys()))
        with rxr.open_rasterio(ds.assets[band].href, chunks=True, lock=True) as img:
            if nodata is None:
                nodata = img.rio.nodata
            if nodata is not None:
                try:
                    img = (img == nodata).astype("uint8")  # nodata-1 data-0
                    img = img.rio.write_nodata(0)
                    if clip is not None:
                        img = clipf(img, clip)  # nodata-1 data-0 outside_nodata-0
                    zeros.append((da.count_nonzero(img) / img.size).compute())
                except Exception:
                    zeros.append(0.0)
            else:
                zeros.append(0.0)
    zerosdict = dict(zip(datasets, zeros, strict=True))
    sortedzeros = dict(sorted(zerosdict.items(), key=lambda item: item[1], reverse=False))
    return list(sortedzeros.keys())
