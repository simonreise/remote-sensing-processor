"""Create patches for DSen2 inference."""

import warnings
from math import ceil

import xbatcher

import dask
import xarray as xr

from remote_sensing_processor.common.common_functions import persist


def get_patches(
    dset: xr.DataArray,
    patch_size: int,
    border: int,
) -> xr.DataArray:
    """Generate patches function itself."""
    # Pad data to make its shape divisible by patch size - border * 2
    shp_pad = []
    for d in dset.shape[1:3]:
        shp_pad.append((patch_size - 2 * border) - (d % (patch_size - 2 * border)))
    dset = dset.pad({"y": (0, shp_pad[0]), "x": (0, shp_pad[1])}, mode="constant", constant_values=0)

    # Generate patches
    batches = xbatcher.BatchGenerator(
        ds=dset,
        input_dims={"x": patch_size - 2 * border, "y": patch_size - 2 * border},
    )

    # Stack patches
    patches = []
    for b in batches:
        patches.append(b)
    patches = xr.concat(patches, "chips", join="override").chunk("auto")

    # Pad data to avoid boundary artifacts
    return patches.pad({"y": (border, border), "x": (border, border)}, mode="symmetric")


def get_test_patches(
    dset_10: xr.DataArray,
    dset_20: xr.DataArray,
    patch_size: int = 128,
    border: int = 8,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Used for inference. Creates patches of specific size in the whole image (10m and 20m)."""
    image_10 = dask.delayed(get_patches)(dset_10, patch_size, border)
    image_20 = dask.delayed(get_patches)(dset_20, patch_size, border)
    image_10, image_20 = dask.compute(image_10, image_20)
    image_10, image_20 = persist(image_10, image_20)
    return image_10, image_20


def get_test_patches60(
    dset_10: xr.DataArray,
    dset_20: xr.DataArray,
    dset_60: xr.DataArray,
    patch_size: int = 192,
    border: int = 12,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Used for inference. Creates patches of specific size in the whole image (10m, 20m and 60m)."""
    image_10 = dask.delayed(get_patches)(dset_10, patch_size, border)
    image_20 = dask.delayed(get_patches)(dset_20, patch_size, border)
    image_60 = dask.delayed(get_patches)(dset_60, patch_size, border)
    image_10, image_20, image_60 = dask.compute(image_10, image_20, image_60)
    image_10, image_20, image_60 = persist(image_10, image_20, image_60)
    return image_10, image_20, image_60


def recompose_images(a: xr.DataArray, border: int, ref: xr.DataArray) -> xr.DataArray:
    """From array with patches recompose the original image."""
    # size = ref.shape

    # # This is done because we do not mirror the data at the image border
    # size = [s - border * 2 for s in size]
    patch_size = a.shape[2] - border * 2

    # print('Patch has dimension {}'.format(patch_size))
    # print('Prediction has shape {}'.format(a.shape))
    # x_tiles = int(ceil(size[2] / float(patch_size)))
    # y_tiles = int(ceil(size[1] / float(patch_size)))
    # print('Tiles per image {} {}'.format(x_tiles, y_tiles))

    # Cutting off the border
    a = a[:, :, border : (a.shape[2] - border), border : (a.shape[3] - border)]

    a = persist(a)
    ref = persist(ref)
    try:
        # a = a.compute()
        ref = ref.compute()
    except Exception:
        warnings.warn("Can not load reference array to memory. Computation could be much slower.", stacklevel=1)

    # Initialize image
    ref = ref.astype(a.dtype)

    for i, prediction in enumerate(a):
        pos1 = patch_size * (i % ceil(ref.shape[1] / patch_size))
        pos2 = patch_size * (i // ceil(ref.shape[1] / patch_size))
        pos3 = min(ref.shape[1], pos1 + patch_size)
        pos4 = min(ref.shape[2], pos2 + patch_size)

        prediction: xr.DataArray

        # Changing prediciton shape if needed
        if (pos3 - pos1) != patch_size:
            prediction = prediction[:, : (pos3 - pos1), :]
        if (pos4 - pos2) != patch_size:
            prediction = prediction[:, :, : (pos4 - pos2)]

        # Writing predicted tile to its position in an array
        ref.data[:, pos1:pos3, pos2:pos4] = prediction

    return ref.chunk("auto")
