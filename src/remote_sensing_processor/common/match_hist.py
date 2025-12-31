"""Histogram matching."""

from pydantic import validate_call
from typing import Optional, Union

import dask
import dask.array as da
import numpy as np
from xarray import DataArray, Dataset

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
def match_hist(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    reference_raster: Union[FilePath, DirectoryPath, PystacItem],
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    nodata: Optional[Union[int, float]] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Matches histograms of two files.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to an input file, directory or a STAC dataset or a STAC Item (e.g., from Planetary Computer).
    reference_raster : string or STAC Item
        A raster that will be used as a reference in histogram matching.
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
    nodata : int or float (default = None)
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
        >>> rsp.match_hist(
        ...     input_path="/home/rsp_test/image_1/sentinel_B1.tif",
        ...     output_path="/home/rsp_test/image_1/sentinel_B1_matched.tif",
        ...     reference_raster="/home/rsp_test/image_2/sentinel_B1.tif",
        ...     nodata=0,
        ... )
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)
    img = load_dataset(dataset)
    img, nodata = prepare_nodata(img, nodata)

    reference_raster = read_dataset(reference_raster)
    ref = load_dataset(reference_raster)

    img = histogram_match(img, ref, nodata)
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


def histogram_match(image: Dataset, reference: Dataset, nodata: Optional[Union[int, float]] = None) -> Dataset:
    """Service function that matches histograms.

    This function is a dask-aware implementation of histogram matching.
    It computes the global histograms of the image and reference arrays without loading the entire data into memory,
    making it suitable for large, chunked dask arrays.
    """
    image = set_nodata_to_nan(image)
    reference = set_nodata_to_nan(reference)

    # Checking if the number of bands in both datasets is the same
    if not len(image.data_vars) == len(reference.data_vars):
        raise ValueError("Datasets have different number of channels")

    # If there is only one data variable, rename it
    if len(image.data_vars) == 1:
        reference = reference.rename({next(iter(list(reference.data_vars))): next(iter(list(image.data_vars)))})

    if not set(image.data_vars) == set(reference.data_vars):
        raise ValueError("Datasets have different bands")

    # Histogram matching
    image = image.map(match_single_band, reference=reference)

    image, _ = prepare_nodata(image, nodata)
    image = restore_nodata_from_nan(image)

    return persist(image)


def match_single_band(image: DataArray, reference: Dataset) -> DataArray:
    """Does a histogram matching for a single band."""
    source_band = image.data
    reference_band = reference[image.name].data

    # Determine the min and max values to define histogram bins, excluding nodata
    vmin = da.nanmin(da.stack([da.nanmin(source_band), da.nanmin(reference_band)]))
    vmax = da.nanmax(da.stack([da.nanmax(source_band), da.nanmax(reference_band)]))
    vmin, vmax = dask.compute(vmin, vmax)

    # Use integer bins for integer data types
    bins = int(vmax - vmin + 1) if np.issubdtype(image.dtype, np.integer) else 256

    # Compute global histograms for source and reference bands
    source_hist, _ = da.histogram(source_band, bins=bins, range=[vmin, vmax])
    reference_hist, _ = da.histogram(reference_band, bins=bins, range=[vmin, vmax])

    # The histograms are small, so we can compute them into memory.
    source_hist, reference_hist = dask.compute(source_hist, reference_hist)

    # Compute Cumulative Distribution Functions (CDFs)
    source_cdf = source_hist.cumsum() / source_hist.sum() if source_hist.sum() > 0 else source_hist
    reference_cdf = reference_hist.cumsum() / reference_hist.sum() if reference_hist.sum() > 0 else reference_hist

    # Compute the lookup table (LUT)
    bin_centers = np.linspace(vmin, vmax, bins)
    interp_values = np.interp(source_cdf, reference_cdf, bin_centers)

    # Apply the LUT to the source band using map_blocks for memory efficiency
    new_data = da.map_blocks(
        lambda block: np.interp(block, bin_centers, interp_values),
        source_band,
        dtype=image.dtype,
    )
    image.data = new_data
    return image


'''
def histogram_match(image, reference, nodata):
    """Service function that matches histograms.

    As match_histograms is a function that is not natively dask-compatible; there are several hacks that try to fix that
    """
    if isinstance(image, xr.Dataset):
        image = image.to_dataarray("band")
        i_ds = True
    else:
        i_ds = False

    if isinstance(reference, xr.Dataset):
        reference = reference.to_dataarray("band")

    # TODO: find a way how to run it without rechunk
    try:
        # If an array consists of several chunks, we need to turn it to one chunk to calc the values for the whole array
        image = image.chunk(-1)
        reference = reference.chunk(-1)
        image.data = dask.array.map_blocks(match_with_nodata, image.data, reference.data, nodata)
        image = image.chunk("auto")
    except Exception:
        # If cannot rechunk, then process blockwise, but the output will contain errors
        warnings.warn(
            "Failed to rechunk the input array to single chunk. The result might be inconsistent.",
            stacklevel=1,
        )
        image.data = dask.array.map_blocks(match_with_nodata, image.data, reference.data, nodata)

    if i_ds:
        image = image.to_dataset("band").expand_dims({"band": 1})

    image, _ = prepare_nodata(image, nodata)

    return image


def match_with_nodata(image, reference, nodata):
    """Performs band-wise histogram matching on masked arrays.

    Adapted from https://gist.github.com/tayden/dcc83424ce55bfb970f60db3d4ddad18
    """
    image_mask = np.where(image == nodata, True, False)
    reference_mask = np.where(reference == nodata, True, False)

    masked_source_image = np.ma.array(image, mask=image_mask)
    masked_reference_image = np.ma.array(reference, mask=reference_mask)

    matched = np.ma.array(np.empty(image.shape, dtype=image.dtype), mask=image_mask, fill_value=nodata)

    for channel in range(masked_source_image.shape[0]):
        matched_channel = match_histograms(
            masked_source_image[channel].compressed(),
            masked_reference_image[channel].compressed(),
        )

        # Re-insert masked background
        mask_ch = image_mask[channel]
        matched[channel][~mask_ch] = matched_channel.ravel()
    return matched.filled()
'''
