"""Generating tiles for change detection."""

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, TypeAdapter, validate_call
from typing import Annotated, Any, Dict, Generator, Optional, Union

import gc
import warnings
from pathlib import Path

import xbatcher

import dask
import numpy as np
import xarray as xr

import datasets

from remote_sensing_processor.common.common_functions import persist, read_json, write_json
from remote_sensing_processor.common.common_raster import assert_equal_shapes
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
    check_classes,
    check_dtype,
    clean_cache,
    create_folders,
    filter_samples,
    get_cache,
    pad,
    prepare_images,
    prepare_raster_sm,
    prepare_vector_sm,
    replace_y_in_meta,
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
    x1: Union[ListOfPath, ListOfPystacItem],
    x2: Union[ListOfPath, ListOfPystacItem],
    y: Union[ListOfDict, None],
    output: NewRSPDS,
    tile_size: Optional[Annotated[int, Field(strict=True, ge=8)]] = 128,
    shuffle: Optional[bool] = False,
    split: Optional[Dict[str, Union[PositiveInt, PositiveFloat]]] = None,
    x_dtype: Optional[DType] = None,
    y_dtype: Optional[DType] = None,
    x_nodata: Optional[Union[int, float]] = None,
    y_nodata: Optional[int] = None,
) -> NewRSPDS:
    """
    Cut rasters into tiles.

    Parameters
    ----------
    x1 : list of paths as strings
        Rasters to use as training data temporal input 1.
    x2 : list of paths as strings
        Rasters to use as training data temporal input 2.
    y : dict or list of dicts (Optional)
        Target variable or multiple target variables. It can be set to None if target value is not needed.
        Dict or multiple dicts. If a target variable is not needed in the dataset, can be set to None.
        It should contain:
        `name`: a name of a target variable that will be used further to call it.
        `path`: raster or vector file to use as target variable.
        `burn_value` (optional): a field to use for a burn-in value. Field should be numeric.
        If there is a `burn_value` key in dict, target variable will be considered a vector file,
        if there is only a `path` key, variable will be considered a raster file.
        We strongly recommend you to change class values to 0, 1, 2, ..., n (where 0 is nodata) before generating tiles.
    output : path as a string
        Path to save generated output x data.
        Data is saved in a .rspds format (custom dataset format based on WebDataset).
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
    y_nodata : int (optional)
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
        >>> x = ["/home/rsp_test/mosaics/sentinel/", "/home/rsp_test/mosaics/dem/dem.tif"]
        >>> y = ["/home/rsp_test/mosaics/landcover.tif", "/home/rsp_test/mosaics/forest_types.tif"]
        >>> out_file = "/home/rsp_test/model/landcover_dataset.rspds"
        >>> out_dataset = rsp.change_detection.generate_tiles(
        ...     x,
        ...     y,
        ...     out_file,
        ...     tile_size=256,
        ...     shuffle=True,
        ...     split={"train": 3, "val": 1, "test": 1},
        ...     x_nodata=0,
        ...     y_nodata=0,
        ... )
        >>> print(out_dataset)
        PosixPath('/home/rsp_test/model/landcover_dataset.rspds')
    """
    if y is not None:
        y = TypeAdapter(ListOfY).validate_python(y)

    if split is None:
        split = {"train": 1}

    unique_id = datasets.fingerprint.Hasher.hash(locals())

    if not len(x1) == len(x2):
        raise ValueError(f"Expected both inputs have the same length, got {len(x1)} and {len(x2)}.")

    if y_nodata is not None and y_nodata != 0:
        warnings.warn(
            "Recommended class values format is 0, 1, 2, ..., n (where 0 is nodata), but y_nodata is " + str(y_nodata),
            stacklevel=1,
        )

    create_folders(output, split)

    data: dict[str, Any] = {
        "task": "change_detection",
    }

    # Initially load and preprocess data
    x1_img, x1_nodata = prepare_images(img=x1, nodata=x_nodata, dtype=x_dtype)
    x2_img, x2_nodata = prepare_images(img=x2, nodata=x_nodata, dtype=x_dtype)

    x_img, x_nodata = construct_bitemporal_input(x1_img, x2_img, x1_nodata, x2_nodata)

    # Write a reference file
    write_reference(x1_img, output, x_nodata)

    if y is not None:
        y_img, y_nodata = prepare_seg_maps(y=y, ref=x_img[0], dtype=y_dtype, y_nodata=y_nodata)
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
            # Checking classes values
            classes = sorted(np.unique(y_img[i]))
            classes = list(range(classes[-1] + 1))
            num_classes = len(classes)
            data["y"][y[i].name]["classes"] = classes
            data["y"][y[i].name]["num_classes"] = num_classes

            if not min(classes) >= 0:
                raise ValueError("Class values must be >= 0")

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
    if y_img is not None:
        samples = filter_samples(y_batches, samples, y_nodata)
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
            "x": datasets.Array4D(dtype="float32", shape=(2, data["x"]["bands"], tile_size, tile_size)),
        }
        if y_img is not None:
            for i in range(len(y)):
                # noinspection PyTypeChecker
                feat["y_" + y[i].name] = datasets.Array2D(dtype="int32", shape=(tile_size, tile_size))
        feat = datasets.Features(feat)

        def dataset_generator(samples: dict[str, list[int]], name: str) -> Generator[dict]:
            for index in samples[name]:
                data_dict = {
                    "key": index,
                    "x": np.pad(
                        x_batches[index].data,
                        ((0, 0), (0, 0), (border, border), (border, border)),
                        "symmetric",
                    ),
                }
                if y is not None:
                    y_index = np.pad(y_batches[index].data, ((0, 0), (border, border), (border, border)), "symmetric")
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


def construct_bitemporal_input(
    x1_img: xr.DataArray,
    x2_img: xr.DataArray,
    x1_nodata: Optional[int],
    x2_nodata: Optional[int],
) -> tuple[xr.DataArray, int]:
    """Merge two segmentation inputs to a bitemporal input."""
    assert_equal_shapes([x1_img, x2_img])
    if x1_nodata != x2_nodata:
        raise ValueError(f"Expected both inputs have the same nodata value, got {x1_nodata} and {x2_nodata}.")
    x_img = xr.concat([x1_img, x2_img], "time")
    return x_img, x1_nodata


def prepare_seg_maps(
    y: ListOfY,
    ref: xr.DataArray,
    dtype: Optional[DType] = None,
    dtype_class: Optional[DType] = np.integer,
    y_nodata: Optional[int] = 0,
) -> tuple[xr.DataArray, int]:
    """Prepare segmentation maps: rasterize vectors and match rasters."""
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
    check_classes(arrays, y_nodata)
    arrays = persist(arrays.squeeze().to_array("band").chunk("auto"))
    return arrays, y_nodata


def load_dataset(dataset: Any) -> tuple[list[Path], Path, dict]:
    """Read dataset files, metadata and reference."""
    # TODO: Replace Any type to DS
    path = dataset.path

    reference = path / "ref.tif"

    meta = read_json(path / "meta.json")

    if meta["task"] != "change_detection":
        raise ValueError("dataset is not a change detection dataset")

    if "y" in meta:
        meta = replace_y_in_meta(meta, dataset)

    files = []
    length = []
    for name in meta["samples"]:
        if dataset.sub == "all" or name in dataset.sub:
            files.append((path / name))
            length.append(len(meta["samples"][name]))
    meta["len"] = length

    return files, reference, meta
