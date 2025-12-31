"""Clip data values."""

from pydantic import validate_call
from typing import Optional, Union

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
from remote_sensing_processor.common.types import DirectoryPath, DType, FilePath, NewPath, PystacItem


@validate_call
def clip_values(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    minimum: Optional[Union[int, float]] = None,
    maximum: Optional[Union[int, float]] = None,
    nodata: Optional[Union[int, float]] = None,
    dtype: Optional[DType] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Clip data values.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
    minimum : int or float (optional)
        Min value.
    maximum : int or float (optional)
        Max value.
    nodata : int or float (default = None)
        Nodata value. If not set, then is read from inputs.
    dtype : dtype definition as a string (optional)
        Requested output data type.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    output_path : pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> # Clip raster values
        >>> rsp.clip_values(
        ...     input_path="/home/rsp_test/sentinel_B1.tif",
        ...     output_path="/home/rsp_test/sentinel_B1_clipped.tif",
        ...     minimum=0,
        ...     maximum=10000,
        ... )
        '/home/rsp_test/sentinel_B1_clipped.json'

        >>> # Clip only lower values
        >>> rsp.clip_values(
        ...     input_path="/home/rsp_test/sentinel_B1.tif",
        ...     output_path="/home/rsp_test/sentinel_B1_clipped.tif",
        ...     minimum=0,
        ... )
        '/home/rsp_test/sentinel_B1_clipped.json'
    """
    if minimum is None and maximum is None:
        raise ValueError("Minimum or maximum should be set")

    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    img, nodata = prepare_nodata(img, nodata)

    # Replacing nodata with nan
    img = set_nodata_to_nan(img)

    img = img.clip(min=minimum, max=maximum)

    # Restoring nodata values
    img = restore_nodata_from_nan(img)

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
