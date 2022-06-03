from math import ceil

from typing import Tuple, List

import numpy as np
from skimage.transform import resize


def interp_patches(image_20: np.ndarray, image_10_shape: Tuple[int, ...]) -> np.ndarray:
    """Upsample patches to shape of higher resolution"""
    data20_interp = np.zeros((image_20.shape[0:2] + image_10_shape[2:4])).astype(
        np.float32
    )
    for k in range(image_20.shape[0]):
        for w in range(image_20.shape[1]):
            data20_interp[k, w] = (
                resize(image_20[k, w] / 30000, image_10_shape[2:4], mode="reflect")
                * 30000
            )  # bilinear
    return data20_interp


def get_patches(
    dset: np.ndarray,
    patch_size: int,
    border: int,
    patches_along_i: int,
    patches_along_j: int,
) -> np.ndarray:
    n_bands = dset.shape[2]

    # array index
    nr_patches = (patches_along_i + 1) * (patches_along_j + 1)
    range_i = np.arange(0, patches_along_i) * (patch_size - 2 * border)
    range_j = np.arange(0, patches_along_j) * (patch_size - 2 * border)

    patches = np.zeros((nr_patches, n_bands) + (patch_size, patch_size)).astype(
        np.float32
    )

    # if height and width are divisible by patch size - border * 2, or if
    # range_i \and range_j are smaller than size
    # add one extra patch at the end of the image
    if (
        np.mod(dset.shape[0] - 2 * border, patch_size - 2 * border) != 0
        or dset.shape[0] - 2 * border / patch_size - 2 * border > patches_along_i
    ):
        range_i = np.append(range_i, (dset.shape[0] - patch_size))
    if (
        np.mod(dset.shape[1] - 2 * border, patch_size - 2 * border) != 0
        or dset.shape[1] - 2 * border / patch_size - 2 * border > patches_along_j
    ):
        range_j = np.append(range_j, (dset.shape[1] - patch_size))

    patch_count = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            upper_left_i = ii
            upper_left_j = jj
            # make shape (p, c, w, h)
            patches[patch_count] = crop_array_to_window(
                dset,
                get_crop_window(upper_left_i, upper_left_j, patch_size, 1),
                rollaxis=True,
            )
            patch_count += 1
    # array shape, ignore unsuscriptable
    # pylint: disable=unsubscriptable-object
    assert patch_count == nr_patches == patches.shape[0]
    return patches


def get_test_patches(
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    patch_size: int = 128,
    border: int = 4,
    interp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Used for inference. Creates patches of specific size in the whole image (10m and 20m)"""

    patch_size_lr = patch_size // 2
    border_lr = border // 2

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = np.pad(
        dset_10,
        ((border, border), (border, border), (0, 0)),
        mode="symmetric",
    )
    dset_20 = np.pad(
        dset_20,
        ((border_lr, border_lr), (border_lr, border_lr), (0, 0)),
        mode="symmetric",
    )

    patches_along_i = (dset_20.shape[0] - 2 * border_lr) // (
        patch_size_lr - 2 * border_lr
    )
    patches_along_j = (dset_20.shape[1] - 2 * border_lr) // (
        patch_size_lr - 2 * border_lr
    )

    image_10 = get_patches(
        dset_10, patch_size, border, patches_along_i, patches_along_j
    )
    image_20 = get_patches(
        dset_20, patch_size_lr, border_lr, patches_along_i, patches_along_j
    )

    image_10_shape = image_10.shape

    if interp:
        data20_interp = interp_patches(image_20, image_10_shape)
    else:
        data20_interp = image_20
    return image_10, data20_interp


# Complex method
# pylint: disable=too-many-locals
def get_test_patches60(
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    dset_60: np.ndarray,
    patch_size: int = 192,
    border: int = 12,
    interp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Used for inference. Creates patches of specific size in the whole image (10m, 20m and 60m)"""

    patch_size_20 = patch_size // 2
    patch_size_60 = patch_size // 6
    border_20 = border // 2
    border_60 = border // 6

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = np.pad(
        dset_10,
        ((border, border), (border, border), (0, 0)),
        mode="symmetric",
    )
    dset_20 = np.pad(
        dset_20,
        ((border_20, border_20), (border_20, border_20), (0, 0)),
        mode="symmetric",
    )
    dset_60 = np.pad(
        dset_60,
        ((border_60, border_60), (border_60, border_60), (0, 0)),
        mode="symmetric",
    )

    patches_along_i = (dset_60.shape[0] - 2 * border_60) // (
        patch_size_60 - 2 * border_60
    )
    patches_along_j = (dset_60.shape[1] - 2 * border_60) // (
        patch_size_60 - 2 * border_60
    )

    image_10 = get_patches(
        dset_10, patch_size, border, patches_along_i, patches_along_j
    )
    image_20 = get_patches(
        dset_20, patch_size_20, border_20, patches_along_i, patches_along_j
    )
    image_60 = get_patches(
        dset_60, patch_size_60, border_60, patches_along_i, patches_along_j
    )

    image_10_shape = image_10.shape

    if interp:
        data20_interp = interp_patches(image_20, image_10_shape)
        data60_interp = interp_patches(image_60, image_10_shape)

    else:
        data20_interp = image_20
        data60_interp = image_60

    return image_10, data20_interp, data60_interp


def get_crop_window(
    upper_left_x: int, upper_left_y: int, patch_size: int, scale: int = 1
) -> List[int]:
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


def crop_array_to_window(
    array: np.ndarray, crop_window: List[int], rollaxis: bool = True
) -> np.ndarray:
    """Return a subset of a numpy array. Rollaxis optional from channels last
    to channels first and vice versa."""
    cropped_array = array[
        crop_window[0] : crop_window[2], crop_window[1] : crop_window[3]
    ]
    if rollaxis:
        return np.rollaxis(
            cropped_array,
            2,
        )
    else:
        return cropped_array


def recompose_images(a: np.ndarray, border: int, size=None) -> np.ndarray:
    """From array with patches recompose original image."""
    if a.shape[0] == 1:
        images = a[0]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a.shape[2] - border * 2

        # print('Patch has dimension {}'.format(patch_size))
        # print('Prediction has shape {}'.format(a.shape))
        x_tiles = int(ceil(size[1] / float(patch_size)))
        y_tiles = int(ceil(size[0] / float(patch_size)))
        # print('Tiles per image {} {}'.format(x_tiles, y_tiles))

        # Initialize image
        # print('Image size is: {}'.format(size))
        images = np.zeros((a.shape[1], size[0], size[1])).astype(np.float32)

        current_patch = 0
        for y in range(0, y_tiles):
            ypoint = y * patch_size
            if ypoint > size[0] - patch_size:
                ypoint = size[0] - patch_size
            for x in range(0, x_tiles):
                xpoint = x * patch_size
                if xpoint > size[1] - patch_size:
                    xpoint = size[1] - patch_size
                images[
                    :, ypoint : ypoint + patch_size, xpoint : xpoint + patch_size
                ] = a[
                    current_patch,
                    :,
                    border : a.shape[2] - border,
                    border : a.shape[3] - border,
                ]
                current_patch += 1

    return images.transpose((1, 2, 0))
