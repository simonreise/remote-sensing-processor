"""Get normalization parameters."""

from pydantic import validate_call
from typing import Optional, Union

import xarray as xr

from remote_sensing_processor.common.common_functions import persist
from remote_sensing_processor.common.common_raster import load_dataset, prepare_nodata, set_nodata_to_nan
from remote_sensing_processor.common.dataset import read_dataset
from remote_sensing_processor.common.types import (
    ListOfPath,
    ListOfPystacItem,
    Percent,
)


@validate_call
def min_max(
    inputs: Union[ListOfPath, ListOfPystacItem],
    nodata: Optional[Union[int, float]] = None,
) -> tuple[float, float]:
    """
    Calculates optimal parameters for min-max normalization.

    Parameters
    ----------
    inputs : string or list of strings or STAC Item or list of STAC Items
        Paths to input files, directories or STAC datasets or STAC Items (e.g., from Planetary Computer).
    nodata : int or float (optional)
        Nodata value. If not set, then is read from inputs.

    Returns
    -------
    minimum : int or float
        Minimum value found in input datasets.
    maximum : int or float
        Minimum value found in input datasets.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.get_normalization_params.min_max(
        ...     "/home/rsp_test/mosaics/sentinel/B1.tif",
        ... )
        (0, 9849)
    """
    mins = []
    maxs = []

    for input_path in inputs:
        dataset = read_dataset(input_path)

        img = load_dataset(dataset)
        img, nodata = prepare_nodata(img, nodata)

        # Replacing nodata with nan
        img = set_nodata_to_nan(img)

        for k in img:
            mins.append(img[k].min(skipna=True).values.item())
            maxs.append(img[k].max(skipna=True).values.item())

        del img

    return min(mins), max(maxs)


@validate_call
def z_score(
    inputs: Union[ListOfPath, ListOfPystacItem],
    nodata: Optional[Union[int, float]] = None,
) -> tuple[float, float]:
    """
    Calculates optimal parameters for z-score normalization.

    Parameters
    ----------
    inputs : string or list of strings or STAC Item or list of STAC Items
        Paths to input files, directories or STAC datasets or STAC Items (e.g., from Planetary Computer).
    nodata : int or float (optional)
        Nodata value. If not set, then is read from inputs.

    Returns
    -------
    mean : int or float
        Mean value for input datasets.
    stddev : int or float
        Mean standard deviation value for input datasets.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.get_normalization_params.z_score(
        ...     "/home/rsp_test/mosaics/sentinel/B1.tif",
        ... )
        (302, 173)
    """
    means = []
    stddevs = []

    for input_path in inputs:
        dataset = read_dataset(input_path)

        img = load_dataset(dataset)
        img, nodata = prepare_nodata(img, nodata)

        # Replacing nodata with nan
        img = set_nodata_to_nan(img)

        for k in img:
            means.append(img[k].mean(skipna=True).values.item())
            stddevs.append(img[k].std(skipna=True).values.item())

        del img

    return sum(means) / len(means), sum(stddevs) / len(stddevs)


@validate_call
def percentile(
    inputs: Union[ListOfPath, ListOfPystacItem],
    percentiles: list[Percent],
    nodata: Optional[Union[int, float]] = None,
) -> dict[int, float]:
    """
    Calculates percentiles for normalization.

    Parameters
    ----------
    inputs : string or list of strings or STAC Item or list of STAC Items
        Paths to input files, directories or STAC datasets or STAC Items (e.g., from Planetary Computer).
    percentiles : list of ints
        Percentiles to be computed. Must be in the 0-100 range.
    nodata : int or float (optional)
        Nodata value. If not set, then is read from inputs.

    Returns
    -------
    percentiles : dict of ints, floats
        Dict where keys are percentiles and values are corresponding percentile values.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.get_normalization_params.percentile(
        ...     "/home/rsp_test/mosaics/sentinel/B1.tif",
        ...     [30, 70],
        ... )
        {30: 129.1, 70: 475.6}
    """
    if percentiles is None or len(percentiles) < 1:
        raise ValueError("Percentiles are not valid.")

    percs = {i: [] for i in percentiles}

    for input_path in inputs:
        dataset = read_dataset(input_path)

        img = load_dataset(dataset)
        img, nodata = prepare_nodata(img, nodata)

        # Replacing nodata with nan
        img = set_nodata_to_nan(img)

        for k in img:
            for p in percentiles:
                percs[p].append(img[k].quantile(p / 100, skipna=True).values.item())

        del img

    return {k: sum(v) / len(v) for k, v in percs.items()}


@validate_call
def dynamic_world(
    inputs: Union[ListOfPath, ListOfPystacItem],
    percentiles: list[Percent],
    nodata: Optional[Union[int, float]] = None,
) -> dict[int, float]:
    """
    Calculates optimal parameters for dynamic world normalization.

    Parameters
    ----------
    inputs : string or list of strings or STAC Item or list of STAC Items
        Paths to input files, directories or STAC datasets or STAC Items (e.g., from Planetary Computer).
    percentiles : list of ints
        Percentiles to be computed. Must be in the 0-100 range.
    nodata : int or float (optional)
        Nodata value. If not set, then is read from inputs.

    Returns
    -------
    percentiles : dict of ints, floats
        Dict where keys are percentiles and values are corresponding percentile values.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> rsp.get_normalization_params.dynamic_world(
        ...     "/home/rsp_test/mosaics/sentinel/B1.tif",
        ...     [30, 70],
        ... )
        {30: 5.49, 70: 5.78}
    """
    if percentiles is None or len(percentiles) < 1:
        raise ValueError("Percentiles are not valid")

    percs = {i: [] for i in percentiles}

    for input_path in inputs:
        dataset = read_dataset(input_path)

        img = load_dataset(dataset)
        img, nodata = prepare_nodata(img, nodata)

        # Replacing nodata with nan
        img = set_nodata_to_nan(img)

        # Log transform
        img = persist(img.where(img == nodata, xr.ufuncs.log(img)))

        for k in img:
            for p in percentiles:
                percs[p].append(img[k].quantile(p / 100, skipna=True).values.item())

        del img

    return {k: sum(v) / len(v) for k, v in percs.items()}
