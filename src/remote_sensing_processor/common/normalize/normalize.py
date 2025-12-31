"""Normalize raster."""

from pydantic import validate_call
from typing import Optional, Union, cast

import warnings

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
    minimum: Optional[Union[int, float]],
    maximum: Optional[Union[int, float]],
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    clip_values: Optional[bool] = False,
    nodata: Optional[Union[int, float]] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Applies min-max normalization to an input file.

    Recommended if you want your data in the 0-1 range.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    minimum : int or float
        Min value.
    maximum : int or float
        Max value.
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
    clip_values: bool (default = False)
        If True, limits output values to min-max range.
        Otherwise, it will just normalize the values and if there are values outside the min-max range,
        they would be larger than 1 or smaller than 0.
    nodata : int or float (optional)
        Nodata value. If not set, then is read from inputs.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.normalize.min_max(
        ...     input_path="/home/rsp_test/mosaics/sentinel/B1.tif",
        ...     minimum=0,
        ...     maximum=10000,
        ...     output_path="/home/rsp_test/mosaics/sentinel/B1_norm.tif",
        ... )
        '/home/rsp_test/mosaics/sentinel/B1_norm.json'
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    img, nodata = prepare_nodata(img, nodata)

    # Replacing nodata with nan
    img = set_nodata_to_nan(img)

    if not minimum < maximum:
        raise ValueError("minimum should be smaller than maximum")

    if img[next(iter(img.keys()))].where(img[next(iter(img.keys()))] != nodata).max().values.item() > maximum:
        warnings.warn(
            "Max value of the data ("
            + str(img[next(iter(img.keys()))].max().values.item())
            + ") is higher than specified max value ("
            + str(maximum)
            + ")",
            stacklevel=1,
        )
    elif img[next(iter(img.keys()))].where(img[next(iter(img.keys()))] != nodata).min().values.item() < minimum:
        warnings.warn(
            "Min value of the data ("
            + str(img[next(iter(img.keys()))].min().values.item())
            + ") is lower than specified min value ("
            + str(minimum)
            + ")",
            stacklevel=1,
        )

    # Normalization
    img = (img - minimum) / (maximum - minimum)
    img = cast("xr.Dataset", cast("object", img))

    # Clipping values
    if clip_values:
        img = img.clip(0, 1)

    # Setting nodata to 0
    img, _ = prepare_nodata(img, 0)
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
    nodata: Optional[Union[int, float]] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """Applies z-score normalization to an input file.

    Recommended if you want all of your data to have the same distribution,
    but the data will not be limited to 0-1 range.

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
    nodata : int or float (optional)
        Nodata value. If not set, then is read from inputs.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.normalize.z_score(
        ...     input_path="/home/rsp_test/mosaics/sentinel/B1.tif",
        ...     mean=302,
        ...     stddev=173,
        ...     output_path="/home/rsp_test/mosaics/sentinel/B1_norm.tif",
        ... )
        '/home/rsp_test/mosaics/sentinel/B1_norm.json'
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    img, nodata = prepare_nodata(img, nodata)

    # Replacing nodata with nan
    img = set_nodata_to_nan(img)

    if img[next(iter(img.keys()))].where(img[next(iter(img.keys()))] != nodata).max().values.item() < mean:
        warnings.warn(
            "Max value of the data ("
            + str(img[next(iter(img.keys()))].max().values.item())
            + ") is lower than specified mean value ("
            + str(mean)
            + ")",
            stacklevel=1,
        )
    elif img[next(iter(img.keys()))].where(img[next(iter(img.keys()))] != nodata).min().values.item() > mean:
        warnings.warn(
            "Min value of the data ("
            + str(img[next(iter(img.keys()))].min().values.item())
            + ") is higher than specified mean value ("
            + str(mean)
            + ")",
            stacklevel=1,
        )

    # Normalization
    img = (img - mean) / stddev
    img = cast("xr.Dataset", cast("object", img))

    # Setting nodata to 0
    img, _ = prepare_nodata(img, 0)
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
    clip_values: Optional[bool] = True,
    nodata: Optional[Union[int, float]] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """Applies log-transform + sigmoid normalization to an input file.

    This normalization method is similar to a method described in https://doi.org/10.1038/s41597-022-01307-4.
    Recommended if you want to have your data in 0-1 range and handle outliers well.

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
    clip_values: bool (default = True)
        If True, will convert negative values in input data to small positive values,
        otherwise negative values will become nodata because ln(neg) is a nan.
    nodata : int or float (optional)
        Nodata value. If not set, then is read from inputs.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.normalize.dynamic_world(
        ...     input_path="/home/rsp_test/mosaics/sentinel/B1.tif",
        ...     percentile1=5.49,
        ...     percentile2=5.78,
        ...     output_path="/home/rsp_test/mosaics/sentinel/B1_norm.tif",
        ... )
        '/home/rsp_test/mosaics/sentinel/B1_norm.json'
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    img, nodata = prepare_nodata(img, nodata)

    # Replacing nodata with nan
    img = set_nodata_to_nan(img)

    if not percentile1 < percentile2:
        raise ValueError("percentile1 should be smaller than percentile2")

    # Normalization
    # Turning 0 to small positive values to log safely
    img = img.clip(1e-05) if clip_values else img.where(img != 0, 1e-05)

    # Log transform
    img = xr.ufuncs.log(img)

    # Linearly rescale log values between percentiles to [-1, 1]
    img = (img - percentile1) / (percentile2 - percentile1)

    # Apply sigmoid to map into (0, 1)
    img = xr.ufuncs.exp(img)
    img = img / (img + 1)

    # Setting nodata to 0
    img, _ = prepare_nodata(img, 0)
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
