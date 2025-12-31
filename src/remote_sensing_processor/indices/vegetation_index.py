"""Calculate vegetation index."""

from pydantic import validate_call
from typing import TYPE_CHECKING, Optional, Union

import spyndex

from remote_sensing_processor.common.common_functions import create_path, persist
from remote_sensing_processor.common.common_raster import (
    check_dtype,
    load_dataset,
    make_nodata_equal,
    prepare_nodata,
    restore_nodata_from_nan,
    set_nodata_to_nan,
    write,
)
from remote_sensing_processor.common.dataset import (
    add_asset,
    add_rsp_metadata,
    check_output,
    postprocess_dataset,
    read_dataset,
)
from remote_sensing_processor.common.types import DirectoryPath, FilePath, NewPath, PystacItem
from remote_sensing_processor.imagery.indices import get_index


if TYPE_CHECKING:
    from xarray import DataArray


@validate_call
def calculate_index(
    name: str,
    input_path: Union[FilePath, DirectoryPath, PystacItem],
    output_path: Optional[Union[DirectoryPath, NewPath]] = None,
    bands: Optional[dict[str, Union[str, int, float]]] = None,
    suffix: Optional[str] = None,
    nodata: Optional[Union[int, float]] = None,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """
    Calculates vegetation indexes.

    Parameters
    ----------
    name : string
        Name of index. If the index is not supported, then it will be considered a normalized difference index.
        RSP supports all the indices supported by `spyndex` and listed
        `here <https://github.com/awesome-spectral-indices/awesome-spectral-indices#spectral-indices-by-application-domain>`_.
    input_path : string or STAC Item
        A path to a directory or a STAC dataset or a STAC Item with imagery product.
        If you define a supported imagery product and a name of supported index, you do not need to define `bands`.
        Bands needed for index calculation are picked automatically.
    output_path : string (optional)
        Path to a directory where the output index will be saved.
        If not set, then will write to the same directory as `input_path`.
        Must be set if input is a remote STAC Item.
    bands: dict of strings, ints or floats (optional)
        Bands and coefficients to calculate the index.
        Only needed if imagery product or the index is not currently supported
        or if you want to set your own custom coefficients or constants values (e.g. C1, C2, g and L for EVI).
        Dictionary keys are the same as
        `here <https://github.com/awesome-spectral-indices/awesome-spectral-indices?tab=readme-ov-file#expressions>`_.
    suffix : str (optional)
        You can define a suffix that will be added to a file name
        if you want the file name of your index to be different from index name
        (e.g. if you want to try several parameter combinations).
    nodata : int or float (default = None)
        Nodata value. If not set, then is read from inputs.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    pathlib.Path
        Path where index raster is saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> # Calculate NDVI from Sentinel 2 product with stac metadata
        >>> ndvi = rsp.calculate_index(
        ...     name="NDVI",
        ...     input_path="/home/rsp_test/mosaics/sentinel/meta.json",
        ... )
        >>> print(ndvi)
        '/home/rsp_test/mosaics/sentinel/NDVI.json'

        >>> # Calculate EVI from Sentinel 2 product by directly defining bands and coefficients
        >>> evi = rsp.calculate_index(
        ...     name="EVI",
        ...     input_path="/home/rsp_test/mosaics/sentinel/",
        ...     output_path="/home/rsp_test/mosaics/sentinel_indices/",
        ...     bands={"N": "B8", "R": "B4", "B": "B2", "g": 2.5, "C1": 6, "C2": 7.5, "L": 1},
        ...     suffix="C1-6",
        ... )
        >>> print(evi)
        '/home/rsp_test/mosaics/sentinel/sentinel_indices/EVI-C1-6.json'
    """
    # Setting output_path if not defined
    output_path = check_output(input_path, output_path, "auto")

    # Setting up a suffix
    suffix = "" if suffix is None else "-" + suffix

    dataset = read_dataset(input_path)

    # Get bands from a product
    bands = get_index(dataset, name, bands)

    # Loading data
    img = load_dataset(dataset, list(bands.values()))
    img, nodata = prepare_nodata(img, nodata)
    img = img.astype("float32")

    # Replacing nodata with nan
    img = set_nodata_to_nan(img)
    img = make_nodata_equal(img)

    # Setting up kernels for kernel indices
    if name == "kEVI":
        bands["kNR"] = spyndex.computeKernel(
            kernel="RBF",
            params={
                "a": img[bands["N"]],
                "b": img[bands["R"]],
                "sigma": (img[bands["N"]].mean() + img[bands["R"]].mean()) / 2,
            },
        )
        bands["kNB"] = spyndex.computeKernel(
            kernel="RBF",
            params={
                "a": img[bands["N"]],
                "b": img[bands["B"]],
                "sigma": (img[bands["N"]].mean() + img[bands["B"]].mean()) / 2,
            },
        )
        bands["kNL"] = spyndex.computeKernel(
            kernel="RBF",
            params={
                "a": img[bands["N"]],
                "b": bands["L"],
                "sigma": (img[bands["N"]].mean() + bands["L"]) / 2,
            },
        )
    elif name == "kIPVI" or name == "kNDVI" or name == "kRVI":
        bands["kNR"] = spyndex.computeKernel(
            kernel="RBF",
            params={
                "a": img[bands["N"]],
                "b": img[bands["R"]],
                "sigma": (img[bands["N"]].mean() + img[bands["R"]].mean()) / 2,
            },
        )
    elif name == "kVARI":
        bands["kGR"] = spyndex.computeKernel(
            kernel="RBF",
            params={
                "a": img[bands["G"]],
                "b": img[bands["R"]],
                "sigma": (img[bands["G"]].mean() + img[bands["R"]].mean()) / 2,
            },
        )
        bands["kGB"] = spyndex.computeKernel(
            kernel="RBF",
            params={
                "a": img[bands["G"]],
                "b": img[bands["B"]],
                "sigma": (img[bands["G"]].mean() + img[bands["B"]].mean()) / 2,
            },
        )

    # Setting up arguments
    bands: dict[str, Union[str, int, float, DataArray]]
    for k, v in bands.items():
        if isinstance(v, str):
            bands[k] = img[v]

    # Calculating an index
    if name in spyndex.indices:
        final = spyndex.computeIndex(index=name, params=bands)
    else:
        # Just computing normalized difference index
        final = spyndex.computeIndex(index="NDVI", params=bands)

    # Setting nodata to 0
    final, _ = prepare_nodata(final, 0)
    # Restoring nodata values
    final = restore_nodata_from_nan(final)

    final.name = name
    final = final.fillna(0)
    final = persist(final)

    final = check_dtype(final)

    # Creating an output folder
    # output_path = pathlib.Path(dataset.get_links("self")[0].href).parent / (name + ".json")
    create_path(output_path)

    # Setting a JSON path
    output_path = output_path / (name + suffix + ".json")

    # Adding a band with the name of a burn value
    add_asset(
        item=dataset,
        name=name,
        path=name + suffix + ".tif",
    )
    add_rsp_metadata(dataset, rsp_type="Undefined")
    # Creating final STAC dataset
    dataset, json_path = postprocess_dataset(
        dataset,
        final.to_dataset(),
        output_path,
        bands=[name],
    )

    # Write
    write(final, json_path.parent / dataset.assets[name].href)

    if write_stac:
        # Writing JSON metadata file
        dataset.save_object(dest_href=json_path.as_posix())
        return json_path
    return output_path
