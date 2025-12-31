"""Generating tiles for regression."""

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, TypeAdapter, validate_call
from typing import Annotated, Any, Dict, Generator, Optional, Union

import gc

import xbatcher

import dask
import numpy as np
import xarray as xr

import datasets

from remote_sensing_processor.common.common_functions import persist, write_json
from remote_sensing_processor.common.types import (
    DirectoryPath,
    DType,
    FilePath,
    ListOfDict,
    ListOfPath,
    ListOfPystacItem,
    NewRSPDS,
    PystacItem,
    SingleOrList,
)
from remote_sensing_processor.segmentation.tiles import (
    border_pad,
    check_dtype,
    clean_cache,
    create_folders,
    filter_samples,
    get_cache,
    pad,
    prepare_images,
    prepare_raster_sm,
    prepare_vector_sm,
    split_samples,
    write_reference,
)


class Y(BaseModel):
    """Y variable class for user input."""

    name: str
    path: Union[FilePath, DirectoryPath, PystacItem]
    burn_value: Optional[str] = None


ListOfY = SingleOrList[Y]


@validate_call
def generate_tiles(
    x: Union[ListOfPath, ListOfPystacItem],
    y: Union[ListOfDict, None],
    output: NewRSPDS,
    tile_size: Optional[Annotated[int, Field(strict=True, ge=8)]] = 128,
    shuffle: Optional[bool] = False,
    split: Optional[Dict[str, Union[PositiveInt, PositiveFloat]]] = None,
    x_dtype: Optional[DType] = None,
    y_dtype: Optional[DType] = None,
    x_nodata: Optional[Union[int, float]] = None,
    y_nodata: Optional[Union[int, float]] = None,
) -> NewRSPDS:
    """
    Cut rasters into tiles.

    Parameters
    ----------
    x : list of paths as strings
        Rasters to use as training data.
    y : dict or list of dicts
        Target variable or multiple target variables. Can be set to None if target value is not needed.
        Dict or multiple dicts.
        It should contain:
        `name`: a name of a target variable that will be used further to call it.
        `path`: raster file to use as target variable.
        `burn_value` (optional): a field to use for a burn-in value. Field should be numeric.
    output : path as a string (optional)
        Path to save generated output x data.
        Data is saved in a .rspds format (custom dataset format based on WebDataset.
    tile_size : int (default = 128)
        Size of tiles to generate (tile_size x tile_size).
    shuffle : bool (default = False)
        Is a random shuffling of samples needed.
    split : dict (optional)
        Splitting data in subsets.
        Is a dict, where keys are the names of split subsets and
        values are numbers defining proportions of every subset.
        For example, `{"train": 3, "validation": 1, "test": 1}` will generate
        3 subsets (train, validation, and test) in proportion 3 to 1 to 1.
    x_dtype : dtype definition as a string (optional)
        If you run out of memory, you can try to convert your data to less memory consuming format.
    y_dtype : dtype definition as a string (optional)
        If you run out of memory, you can try to convert your data to less memory consuming format.
    x_nodata : int or float (optional)
        You can define which value in x raster corresponds to nodata
        and areas that contain nodata in x raster will be ignored while training and testing.
        Tiles that contain only nodata in both x and y will be omitted.
        If not defined, then the most common nodata value amongst x files will be used.
        If there are no nodata values, will be set to 0.
    y_nodata : int or float (optional)
        You can define which value will be used to fill nodata.
        If there are polygons with the same value as `y_nodata`, they will be ignored while training and testing.
        Tiles that contain only nodata in both x and y will be omitted.
        If not defined, then it will be set to 0.

    Returns
    -------
    pathlib.Path
        Path to the output dataset.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> x = ["/home/rsp_test/mosaics/sentinel/sentinel.json", "/home/rsp_test/mosaics/dem/dem.tif"]
        >>> y = [
        ...     {"name": "nitrogen", "path": "/home/rsp_test/mosaics/nitrogen.tif"},
        ...     {"name": "phosphorus", "path": "/home/rsp_test/vectors/phosphorus.gpkg", "burn_value": "P"},
        ... ]
        >>> out_file = "/home/rsp_test/model/chem_dataset.rspds"
        >>> out_dataset = rsp.regression.generate_tiles(
        ...     x,
        ...     y,
        ...     out_file,
        ...     tile_size=256,
        ...     shuffle=True,
        ...     split={"train": 3, "val": 1, "test": 1},
        ... )
        >>> print(out_dataset)
        PosixPath('/home/rsp_test/model/chem_dataset.rspds')
    """
    if y is not None:
        y = TypeAdapter(ListOfY).validate_python(y)

    if split is None:
        split = {"train": 1}

    unique_id = datasets.fingerprint.Hasher.hash(locals())

    create_folders(output, split)

    data: dict[str, Any] = {
        "task": "regression",
    }

    # Initially load and preprocess data
    x_img, x_nodata = prepare_images(img=x, nodata=x_nodata, dtype=x_dtype)

    # Write a reference file
    write_reference(x_img, output, x_nodata)

    if y is not None:
        y_img, y_nodata = prepare_seg_maps(y=y, y_nodata=y_nodata, ref=x_img[0], dtype=y_dtype, dtype_class=np.floating)
        # Masking areas where y_img is not nodata, but x_img is nodata
        y_img = y_img.where(x_img[0] != x_nodata, y_nodata)
        # x_img = x_img.where(y_img[0] != y_nodata, x_nodata)
        if x_img.shape[1:] != y_img.shape[1:]:
            raise ValueError("x and y have different shapes")
    else:
        y_img, y_nodata = None, None

    border, padding = border_pad(x_img, tile_size)
    data["tile_size"] = tile_size
    data["border"] = border
    data["pad"] = padding
    # Padding
    x_img = pad(x_img, padding, x_nodata)
    if y_img is not None:
        y_img = pad(y_img, padding, y_nodata)

    data["x"] = {
        "dtype": x_img.dtype,
        "nodata": x_nodata,
        "bands": x_img.shape[0],
        "variables": ["_".join(x.split("_")[1:]) for x in x_img["band"].values.tolist()],
    }

    if y_img is not None:
        data["y"] = {}
        for i in range(len(y_img)):
            # noinspection PyUnresolvedReferences
            data["y"][y[i].name] = {"dtype": y_img[i].dtype, "nodata": y_nodata}

    # Setting up tiles generators
    x_batches = xbatcher.BatchGenerator(
        ds=x_img,
        input_dims={"x": tile_size - (border * 2), "y": tile_size - (border * 2)},
    )
    if y_img is not None:
        y_batches = xbatcher.BatchGenerator(
            ds=y_img,
            input_dims={"x": tile_size - (border * 2), "y": tile_size - (border * 2)},
        )
    else:
        y_batches = None

    # Getting samples
    samples = list(range(len(x_batches)))
    samples = filter_samples(x_batches, samples, x_nodata)
    # Shuffling samples
    if shuffle:
        np.random.shuffle(samples)
    # Splitting samples
    samples = split_samples(samples, split)
    data["samples"] = samples

    write_json(data, output / "meta.json")

    for name in split:
        # Generate features
        feat = {
            "key": datasets.Value(dtype="int64"),
            "x": datasets.Array3D(dtype="float32", shape=(data["x"]["bands"], tile_size, tile_size)),
        }
        if y_img is not None:
            for i in range(len(y)):
                # noinspection PyTypeChecker
                feat["y_" + y[i].name] = datasets.Array2D(dtype="float32", shape=(tile_size, tile_size))
        feat = datasets.Features(feat)

        def dataset_generator(samples: dict[str, list[int]], name: str) -> Generator[dict]:
            for index in samples[name]:
                data_dict = {
                    "key": index,
                    "x": np.pad(
                        x_batches[index].data,
                        ((0, 0), (border, border), (border, border)),
                        "symmetric",
                    ),
                }
                if y is not None:
                    y_index = np.pad(
                        y_batches[index].data,
                        ((0, 0), (border, border), (border, border)),
                        "symmetric",
                    )
                    for j in range(len(y)):
                        data_dict["y_" + y[j].name] = y_index[j]
                yield data_dict

        # Create dataset
        ds = datasets.Dataset.from_generator(
            dataset_generator,
            features=feat,
            cache_dir=(output / ".cache").as_posix(),
            fingerprint=unique_id,
            gen_kwargs={"samples": samples, "name": name},
        )

        # Save dataset
        ds.save_to_disk(output / name)

        # Cleaning the cache
        cache = get_cache(ds)
        del ds
        gc.collect()
        clean_cache(cache)
    return output


def prepare_seg_maps(
    y: ListOfY,
    ref: xr.DataArray,
    dtype: Optional[type] = None,
    dtype_class: Optional[type] = np.floating,
    y_nodata: Optional[Union[int, float]] = 0,
) -> tuple[xr.DataArray, Union[int, float]]:
    """Prepare segmentation maps: match rasters."""
    if y_nodata is None:
        y_nodata = 0
    arrays = []
    for ds in y:
        if ds.burn_value is not None:
            arrays.append(dask.delayed(prepare_vector_sm)(ds.path, ref, ds.burn_value, ds.name, y_nodata))
        else:
            arrays.append(dask.delayed(prepare_raster_sm)(ds.path, ref, ds.name, y_nodata))
    arrays = list(dask.compute(*arrays))
    arrays = xr.merge(arrays)
    arrays = check_dtype(img=arrays, dtype=dtype, dtype_class=dtype_class)
    arrays = persist(arrays.squeeze().to_array("band").chunk("auto"))
    return arrays, y_nodata
