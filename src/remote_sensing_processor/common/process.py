"""General and simple raster processing in one function."""

from pydantic import PositiveInt, validate_call
from typing import Optional, Union

import xarray as xr

from remote_sensing_processor.common.common_functions import create_path, persist
from remote_sensing_processor.common.common_raster import (
    check_dtype,
    clipf,
    load_dataset,
    prepare_nodata,
    reproject,
    reproject_match,
    write_dataset,
)
from remote_sensing_processor.common.dataset import check_output, postprocess_dataset, read_dataset
from remote_sensing_processor.common.fill import fillnodata
from remote_sensing_processor.common.types import CRS, DirectoryPath, DType, FilePath, NewPath, PystacItem


@validate_call
def process(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    fill_nodata: Optional[bool] = False,
    fill_distance: Optional[PositiveInt] = 250,
    clip: Optional[FilePath] = None,
    crs: Optional[CRS] = None,
    nodata: Optional[Union[int, float]] = None,
    reference_raster: Optional[Union[FilePath, DirectoryPath, PystacItem]] = None,
    resample: Optional[str] = "average",
    dtype: Optional[DType] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Processes a single raster. Can clip, reproject, match, fill nodata and change data type.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
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
    dtype : dtype definition as a string (optional)
        Requested output data type.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> # Fill gaps in a raster
        >>> rsp.process(
        ...     input_path="/home/rsp_test/sentinel_B1.tif",
        ...     output_path="/home/rsp_test/sentinel_B1_filled.tif",
        ...     fill_nodata=True,
        ...     fill_distance=500,
        ... )
        '/home/rsp_test/sentinel_B1_filled.json'

        >>> # Clip and reproject Sentinel-2 dataset
        >>> rsp.process(
        ...     input_path="/home/rsp_test/sentinels/Sentinel1/meta.json",
        ...     clip="/home/rsp_test/site_border.gpkg",
        ...     crs="EPSG:4269",
        ... )
        '/home/rsp_test/sentinels/Sentinel1/meta.json'

        >>> # Match a raster with a reference raster
        >>> rsp.process(
        ...     input_path="/home/rsp_test/DEM.tif",
        ...     output_path="/home/rsp_test/DEM_matched.tif",
        ...     reference_raster="/home/rsp_test/sentinel_B1.tif",
        ... )
        '/home/rsp_test/DEM_matched.json'
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset, clip=clip)
    img, nodata = prepare_nodata(img, nodata)

    # Reproject
    if crs is not None:
        img = reproject(img, crs)

    # Matching a raster to another raster
    if reference_raster is not None:
        reference_raster = read_dataset(reference_raster)
        ref = load_dataset(reference_raster)
        img = reproject_match(img, ref, resample)

    # Clip
    if clip is not None:
        img = clipf(img, clip)

    # Fill nodata
    if fill_nodata:
        mask = xr.ones_like(img)
        # nodata -> 0, data -> 1
        mask = mask.where(img != nodata, 0)
        # outside -> 1
        for band in mask:
            mask[band] = mask[band].rio.write_nodata(1)
        mask = clipf(mask, clip)
        # Fill nodata
        img = fillnodata(img, mask, fill_distance, nodata)

    # Changing dtype
    if dtype:
        img = img.astype(dtype)
        img = persist(img)

    img = check_dtype(img)

    # Creating an output folder
    create_path(output_path)

    # Creating final STAC dataset
    dataset, json_path = postprocess_dataset(dataset, img, output_path)

    # Write
    write_dataset(img, dataset, json_path)

    if write_stac:
        # Writing JSON metadata file
        dataset.save_object(dest_href=json_path.as_posix())
        return json_path
    return output_path
