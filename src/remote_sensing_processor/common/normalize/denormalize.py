"""Denormalize raster."""

from pydantic import validate_call
from typing import Optional, Union, cast

import xarray as xr

from remote_sensing_processor.common.common_functions import create_path, persist
from remote_sensing_processor.common.common_raster import (
    check_dtype,
    load_dataset,
    prepare_nodata,
    restore_nodata_from_nan,
    set_nodata_to_nan,
    write_dataset,
)
from remote_sensing_processor.common.dataset import check_output, postprocess_dataset, read_dataset
from remote_sensing_processor.common.types import DirectoryPath, FilePath, NewPath, PystacItem


@validate_call
def min_max(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    minimum: Optional[Union[int, float]] = None,
    maximum: Optional[Union[int, float]] = None,
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    nodata: Optional[Union[int, float]] = 0,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Recovers original values from min-max normalized raster.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    minimum: int or float
        Min value that was used for normalization.
    maximum: int or float
        Max value that was used for normalization.
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
    nodata : int or float (default = 0)
        Nodata value to be used in output data. If not set, then is read from inputs.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.denormalize.min_max(
        ...     input_path="/home/rsp_test/mosaics/sentinel/B1_norm.tif",
        ...     minimum=0,
        ...     maximum=10000,
        ...     output_path="/home/rsp_test/mosaics/sentinel/B1_orig.tif",
        ... )
        '/home/rsp_test/mosaics/sentinel/B1_orig.json'
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    img, _ = prepare_nodata(img, 0)

    # Replacing nodata with nan
    img = set_nodata_to_nan(img)

    if not minimum < maximum:
        raise ValueError("minimum should be smaller than maximum")

    # Denormalize
    img = (img * (maximum - minimum)) + minimum
    img = cast("xr.Dataset", cast("object", img))

    # Setting nodata to desired value
    img, nodata = prepare_nodata(img, nodata)
    # Restoring nodata values
    img = restore_nodata_from_nan(img)

    img = check_dtype(img)
    img = persist(img)

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


@validate_call
def z_score(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    mean: Optional[Union[int, float]],
    stddev: Optional[Union[int, float]],
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    nodata: Optional[Union[int, float]] = 0,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Recovers original values from z-score normalized raster.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    mean : int or float
        Mean value.
    stddev : int or float
        Standard deviation value.
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
    nodata : int or float (default = 0)
        Nodata value to be used in output data. If not set, then is read from inputs.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.denormalize.z_score(
        ...     input_path="/home/rsp_test/mosaics/sentinel/B1_norm.tif",
        ...     mean=302,
        ...     stddev=173,
        ...     output_path="/home/rsp_test/mosaics/sentinel/B1_orig.tif",
        ... )
        '/home/rsp_test/mosaics/sentinel/B1_orig.json'
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    img, _ = prepare_nodata(img, 0)

    # Replacing nodata with nan
    img = set_nodata_to_nan(img)

    # Denormalize
    img = img * stddev + mean
    img = cast("xr.Dataset", cast("object", img))

    # Setting nodata to desired value
    img, nodata = prepare_nodata(img, nodata)
    # Restoring nodata values
    img = restore_nodata_from_nan(img)

    img = check_dtype(img)
    img = persist(img)

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


@validate_call
def dynamic_world(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    percentile1: Optional[Union[int, float]],
    percentile2: Optional[Union[int, float]],
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    nodata: Optional[Union[int, float]] = 0,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Recovers original values from dynamic world normalized raster.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    percentile1 : int or float
        First log-transformed data percentile.
    percentile2 : int or float
        Second log-transformed data percentile.
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
    nodata : int or float (default = 0)
        Nodata value to be used in output data. If not set, then is read from inputs.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.denormalize.dynamic_world(
        ...     input_path="/home/rsp_test/mosaics/sentinel/B1_norm.tif",
        ...     percentile1=5.49,
        ...     percentile2=5.78,
        ...     output_path="/home/rsp_test/mosaics/sentinel/B1_orig.tif",
        ... )
        '/home/rsp_test/mosaics/sentinel/B1_orig.json'
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    img, _ = prepare_nodata(img, 0)

    # Replacing nodata with nan
    img = set_nodata_to_nan(img)

    if not percentile1 < percentile2:
        raise ValueError("percentile1 should be smaller than percentile2")

    # Inverting sigmoid
    img = img / (1 - img)
    img = xr.ufuncs.log(img)

    # Inverting linear rescaling
    img = (img * (percentile2 - percentile1)) + percentile1

    # Inverting log transform
    img = xr.ufuncs.exp(img)

    # Setting nodata to desired value
    img, nodata = prepare_nodata(img, nodata)
    # Restoring nodata values
    img = restore_nodata_from_nan(img)

    img = check_dtype(img)
    img = persist(img)

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
