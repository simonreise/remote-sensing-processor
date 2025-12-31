"""Replace value or nodata in a raster file."""

from pydantic import validate_call
from typing import Optional, Union

import numpy as np

from pystac import Item

from remote_sensing_processor.common.common_functions import create_path, persist
from remote_sensing_processor.common.common_raster import (
    check_dtype,
    load_dataset,
    prepare_nodata,
    write_dataset,
)
from remote_sensing_processor.common.dataset import check_output, postprocess_dataset, read_dataset
from remote_sensing_processor.common.types import DirectoryPath, FilePath, NewPath, PystacItem


@validate_call
def replace_value(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    values: Optional[dict[Union[int, float], Union[int, float]]] = None,
    old: Optional[Union[int, float]] = None,
    new: Optional[Union[int, float]] = None,
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Replaces specific values in a raster.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    values: dict of ints or floats (optional)
        Mapping from old values to new values.
    old: int or float (optional)
        An old value to replace.
    new: int or float (optional)
        A new value to insert.
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.replace_value(
        ...     input_path="/home/rsp_test/mosaics/sentinel/B1.tif",
        ...     output_path="/home/rsp_test/mosaics/sentinel/B1_new.tif",
        ...     old=0,
        ...     new=-9999,
        ... )

        >>> rsp.replace_value(
        ...     input_path="/home/rsp_test/replace/WorldCover.tif",
        ...     output_path="/home/rsp_test/replace/WorldCover_replaced.tif",
        ...     values={10: 1, 20: 1, 30: 2, 40: 2, 50: 4, 60: 4, 70: 3, 80: 2, 90: 1, 100: 2},
        ... )

    """
    if values is None and (old is None and new is None):
        raise ValueError("values or old and new must be defined")
    if (old is not None and new is None) or (old is None and new is not None):
        raise ValueError("Both old and new must be set")
    return replace_val(
        input_path=input_path,
        output_path=output_path,
        values=values,
        new=new,
        old=old,
        nodata=False,
        write_stac=write_stac,
    )


@validate_call
def replace_nodata(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    new: Union[int, float],
    old: Optional[Union[int, float]] = None,
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Replaces a nodata value in a raster.

    Parameters
    ----------
    input_path : string or STAC Item
        Path to input file, directory or STAC dataset or a STAC Item (e.g., from Planetary Computer).
    new: int or float
        A new nodata value to insert.
    old: int or float (optional)
        An old nodata value to replace. If not set, then is read from inputs.
    output_path : string (optional)
        Path to an output file, directory, or STAC dataset. If it is not set, then will overwrite the input files.
        Must be set if input is a remote STAC Item.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where output raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.replace_nodata(
        ...     input_path="/home/rsp_test/mosaics/landcover/landcover.tif",
        ...     output_path="/home/rsp_test/mosaics/landcover/landcover_new.tif",
        ...     new=0,
        ... )
    """
    return replace_val(
        input_path=input_path,
        output_path=output_path,
        values=None,
        new=new,
        old=old,
        nodata=True,
        write_stac=write_stac,
    )


def replace_val(
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    output_path: Optional[Union[FilePath, DirectoryPath, NewPath]] = None,
    values: Optional[dict[Union[int, float], Union[int, float]]] = None,
    new: Optional[Union[int, float]] = None,
    old: Optional[Union[int, float]] = None,
    nodata: bool = False,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """Replace value function itself."""
    output_path = check_output(input_path, output_path)

    dataset = read_dataset(input_path)

    img = load_dataset(dataset)
    if nodata:
        img, old = prepare_nodata(img, old)
        if old is None:
            raise ValueError('Input file does not contain nodata value. Please set it explicitly with "old" arg')

    # Replacing multiple values
    if values is not None:
        # Changing dtype
        dtype = next(img[b].dtype.name for b in img)
        if np.issubdtype(dtype, np.integer):
            img = img.astype("int64")
        elif np.issubdtype(dtype, np.floating):
            img = img.astype("float64")

        # Replacing values
        for k, v in values.items():
            img = img.where(img != k, v + 100000)
        for _, v in values.items():
            img = img.where(img != v + 100000, img - 100000)

        # Restoring datatype
        img = img.astype(dtype)
    else:
        # Replacing a single value
        img = img.where(img != old, new)
    img = check_dtype(img)
    img = persist(img)

    # Rewrite nodata
    if nodata:
        img, _ = prepare_nodata(img, new)

    # Creating an output folder
    create_path(output_path)

    # Creating final STAC dataset
    dataset = postprocess_classification(dataset, values, new, old)
    dataset, json_path = postprocess_dataset(dataset, img, output_path)

    # Write
    write_dataset(img, dataset, json_path)

    if write_stac:
        # Writing JSON metadata file
        dataset.save_object(dest_href=json_path.as_posix())
        return json_path
    return output_path


def postprocess_classification(
    stac: Item,
    values: Optional[dict[Union[int, float], Union[int, float]]] = None,
    new: Optional[Union[int, float]] = None,
    old: Optional[Union[int, float]] = None,
) -> Item:
    """Replacing values in Classification STAC extension metadata."""
    if new is not None and old is not None:
        values = {old: new}

    for asset in stac.assets:
        # Replacing values in Classification STAC extension metadata
        if "classification:classes" in stac.assets[asset].extra_fields:
            for k, v in values.items():
                for i in range(len(stac.assets[asset].extra_fields["classification:classes"])):
                    if stac.assets[asset].extra_fields["classification:classes"][i]["value"] == k:
                        stac.assets[asset].extra_fields["classification:classes"][i]["value"] = v + 100000
            for _, v in values.items():
                for i in range(len(stac.assets[asset].extra_fields["classification:classes"])):
                    if stac.assets[asset].extra_fields["classification:classes"][i]["value"] == v + 100000:
                        stac.assets[asset].extra_fields["classification:classes"][i]["value"] = v
        # Cannot work with bitfields, removing extension
        if "classification:bitfields" in stac.assets[asset].extra_fields:
            del stac.assets[asset].extra_fields["classification:bitfields"]
            if "https://stac-extensions.github.io/classification/v2.0.0/schema.json" in stac.stac_extensions:
                stac.stac_extensions.remove("https://stac-extensions.github.io/classification/v2.0.0/schema.json")
    return stac
