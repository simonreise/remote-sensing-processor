"""Calculate curvature from a DEM."""

from pydantic import validate_call
from typing import Optional, Union

import xarray as xr

import xrspatial

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
from remote_sensing_processor.common.fill import fillnodata
from remote_sensing_processor.common.types import DirectoryPath, FilePath, NewPath, PystacItem


@validate_call
def curvature(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    normalize: Optional[bool] = False,
    nodata: Optional[Union[int, float]] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """Calculates the curvature (second derivative) of each cell based on the elevation of its neighbors in a 3x3 grid.

    Positive curvature indicates the surface is upwardly convex. A negative value indicates it is upwardly concave.
    A value of 0 indicates a flat surface.


    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
    normalize:
        Whether min-max data normalization needed.
    nodata : int or float (default = None)
        Nodata value. If not set, then is read from inputs.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    output_path : pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.dem.curvature(
        ...     input_path="/home/rsp_test/DEM.tif",
        ...     output_path="/home/rsp_test/curvature.tif",
        ... )
    """
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    img, nodata = prepare_nodata(img, nodata, -9999)

    # First we fill nodata with nan
    img = set_nodata_to_nan(img)

    # Calculating curvature
    product = persist(img.map(xrspatial.curvature))

    # Filling nodata values that appeared on image borders and on nodata areas border
    mask = xr.where((img.notnull()) & (product.isnull()), 0, 1)
    product = fillnodata(product, mask, 3, nodata)

    # Normalizing data if needed
    if normalize:
        product = product.where(product == nodata, (product - -4) / (4 - -4))
        # Generally curvature should be in range [-4, 4]. If it is not, probably, it is an error.
        product = product.where(product <= 1, 1)
        product = product.where((product >= 0) | (product == nodata), 0)
        # Setting nodata to 0
        product, _ = prepare_nodata(product, 0)

    # Restoring nodata values
    product = restore_nodata_from_nan(product)

    product = check_dtype(product)
    img = persist(product)

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
