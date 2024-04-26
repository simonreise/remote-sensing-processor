from math import ceil
from functools import partial

import numpy as np
import dask
import xarray
from skimage.transform import rescale

from remote_sensing_processor.common.common_functions import persist


def interp_patch(patch, scale):
    return xarray.apply_ufunc(
        rescale, 
        patch.astype('float32'),
        input_core_dims=[['y', 'x']],
        output_core_dims=[['y', 'x']], 
        exclude_dims={'y', 'x'},
        dask='parallelized', 
        dask_gufunc_kwargs={'output_sizes': {'y': patch.y.shape[0] * scale, 'x': patch.x.shape[0] * scale}},
        kwargs={'scale': scale, 'mode': 'reflect', 'channel_axis': 0},
    )


def get_patches(
    dset,
    patch_size,
    border,
    patches_along_i,
    patches_along_j,
):
    n_bands = dset.shape[2]

    # Array index
    nr_patches = (patches_along_i + 1) * (patches_along_j + 1)
    range_i = np.arange(0, patches_along_i) * (patch_size - 2 * border)
    range_j = np.arange(0, patches_along_j) * (patch_size - 2 * border)

    # If height and width are divisible by patch size - border * 2, or if
    # range_i \and range_j are smaller than size
    # add one extra patch at the end of the image
    if (
        (dset.shape[1] - 2 * border) % (patch_size - 2 * border) != 0
        or dset.shape[1] - 2 * border / patch_size - 2 * border > patches_along_i
    ):
        range_i = np.append(range_i, (dset.shape[1] - patch_size))
    if (
        (dset.shape[2] - 2 * border) % (patch_size - 2 * border) != 0
        or dset.shape[2] - 2 * border / patch_size - 2 * border > patches_along_j
    ):
        range_j = np.append(range_j, (dset.shape[2] - patch_size))

    stack = []
    patch_count = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            upper_left_i = ii
            upper_left_j = jj
            window = get_crop_window(upper_left_i, upper_left_j, patch_size, 1)
            stack.append(dset.isel(y=slice(window[0], window[2]), x=slice(window[1], window[3])))
            patch_count += 1
    patches = xarray.concat(stack, 'chips', join="override").chunk('auto')
    # Array shape, ignore unsuscriptable
    assert patch_count == nr_patches == patches.shape[0]
    return patches


def get_test_patches(
    dset_10,
    dset_20,
    patch_size=128,
    border=4,
    interp=True
):
    """Used for inference. Creates patches of specific size in the whole image (10m and 20m)"""

    patch_size_20 = patch_size // 2
    border_20 = border // 2

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = dset_10.pad({'x': border, 'y': border}, mode="symmetric")
    dset_20 = dset_20.pad({'x': border_20, 'y': border_20}, mode="symmetric")
    dset_10, dset_20 = persist(dset_10, dset_20)

    patches_along_i = (dset_20.shape[1] - 2 * border_20) // (patch_size_20 - 2 * border_20)
    patches_along_j = (dset_20.shape[2] - 2 * border_20) // (patch_size_20 - 2 * border_20)

    image_10 = dask.delayed(get_patches)(dset_10, patch_size, border, patches_along_i, patches_along_j)
    image_20 = dask.delayed(get_patches)(dset_20, patch_size_20, border_20, patches_along_i, patches_along_j)
    image_10, image_20 = dask.compute(image_10, image_20)
    image_10, image_20 = persist(image_10, image_20)

    if interp:
        coef20 = int(image_10.shape[2] / image_20.shape[2]) 
        rescale_p = partial(interp_patch, scale=coef20)
        data20_interp = image_20.groupby('chips', squeeze=False).map(rescale_p)
    else:
        data20_interp = image_20
    image_10, data20_interp = persist(image_10, data20_interp)
    return image_10, data20_interp


def get_test_patches60(
    dset_10,
    dset_20,
    dset_60,
    patch_size=192,
    border=12,
    interp=True
):
    """Used for inference. Creates patches of specific size in the whole image (10m, 20m and 60m)"""

    patch_size_20 = patch_size // 2
    patch_size_60 = patch_size // 6
    border_20 = border // 2
    border_60 = border // 6

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = dset_10.pad({'x': border, 'y': border}, mode="symmetric")
    dset_20 = dset_20.pad({'x': border_20, 'y': border_20}, mode="symmetric")
    dset_60 = dset_60.pad({'x': border_60, 'y': border_60}, mode="symmetric")
    dset_10, dset_20, dset_60 = persist(dset_10, dset_20, dset_60)

    patches_along_i = (dset_60.shape[1] - 2 * border_60) // (patch_size_60 - 2 * border_60)
    patches_along_j = (dset_60.shape[2] - 2 * border_60) // (patch_size_60 - 2 * border_60)

    image_10 = dask.delayed(get_patches)(dset_10, patch_size, border, patches_along_i, patches_along_j)
    image_20 = dask.delayed(get_patches)(dset_20, patch_size_20, border_20, patches_along_i, patches_along_j)
    image_60 = dask.delayed(get_patches)(dset_60, patch_size_60, border_60, patches_along_i, patches_along_j)
    image_10, image_20, image_60 = dask.compute(image_10, image_20, image_60)
    image_10, image_20, image_60 = persist(image_10, image_20, image_60)

    if interp:
        coef20 = int(image_10.shape[2] / image_20.shape[2]) 
        rescale_p = partial(interp_patch, scale=coef20)
        data20_interp = image_20.groupby('chips', squeeze=False).map(rescale_p)
        coef60 = int(image_10.shape[2] / image_60.shape[2]) 
        rescale_p = partial(interp_patch, scale=coef60)
        data60_interp = image_60.groupby('chips', squeeze=False).map(rescale_p)

    else:
        data20_interp = image_20
        data60_interp = image_60

    image_10, data20_interp, data60_interp = persist(image_10, data20_interp, data60_interp)

    return image_10, data20_interp, data60_interp


def get_crop_window(upper_left_x, upper_left_y, patch_size, scale=1):
    """From a x,y coordinate pair and patch size return a list ofpixel coordinates
    defining a window in an array. Optionally pass a scale factor."""
    crop_window = [
        upper_left_x,
        upper_left_y,
        upper_left_x + patch_size,
        upper_left_y + patch_size,
    ]
    crop_window = [p * scale for p in crop_window]
    return crop_window


def recompose_images(a, border, ref):
    """From array with patches recompose original image."""
    size = ref.shape
    if a.shape[0] == 1:
        images = a[0]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a.shape[2] - border * 2

        # print('Patch has dimension {}'.format(patch_size))
        # print('Prediction has shape {}'.format(a.shape))
        x_tiles = int(ceil(size[2] / float(patch_size)))
        y_tiles = int(ceil(size[1] / float(patch_size)))
        # print('Tiles per image {} {}'.format(x_tiles, y_tiles))

        # Initialize image
        # print('Image size is: {}'.format(size))
        # TODO : uses compute because of notimplementederror: xarray can't set arrays with multiple array indices to dask yet
        images = xarray.concat(
            [xarray.full_like(ref[0], 0, dtype='float32') for i in range(a.shape[1])],
            dim='band',
        ).compute()
        current_patch = 0
        for y in range(0, y_tiles):
            ypoint = y * patch_size
            if ypoint > size[1] - patch_size:
                ypoint = size[1] - patch_size
            for x in range(0, x_tiles):
                xpoint = x * patch_size
                if xpoint > size[2] - patch_size:
                    xpoint = size[2] - patch_size
                # TODO: not working with dask arrays, so had to load reference without chunking, can be memory-consuming.
                area = images.isel(y=slice(ypoint, ypoint + patch_size), x=slice(xpoint, xpoint + patch_size))
                images.loc[{'x': area.x, 'y': area.y}] = a[
                    current_patch,
                    :,
                    border : a.shape[2] - border,
                    border : a.shape[3] - border,
                ]
                current_patch += 1
    return images.chunk('auto')