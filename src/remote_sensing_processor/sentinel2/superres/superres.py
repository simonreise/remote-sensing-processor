import os
import gc
import re
from pathlib import Path
import glob
import warnings

import xarray
import dask

import rasterio
import rioxarray
import geopandas as gpd

from remote_sensing_processor.common.common_functions import persist, convert_3D_2D

from remote_sensing_processor.sentinel2.superres.supres import dsen2_20, dsen2_60

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# This code is adapted from this repository
# https://github.com/lanha/DSen2 and is distributed under the same
# license.


def superresolution(input_dir="/tmp/input/", clip=None):
    """
    This function takes the raster data at 10, 20, and 60 m resolutions and by applying
    data_final method creates the input data for the convolutional neural network.
    It returns 10 m resolution for all the bands in 20 and 60 m resolutions.
    """
    # Reading datasets
    datasets, image_level = get_data(input_dir)

    # Reading data from each dataset
    futures = []
    for dataset in datasets:
        futures.append(dask.delayed(read_ds)(dataset))
    (data10, dic_10m), (data20, dic_20m), (data60, dic_60m) = dask.compute(*futures)
    data10, data20, data60 = persist(data10, data20, data60)
    
    # Clipping
    if clip is not None:
        data60, box = get_clip(data60, clip)
        data10 = dask.delayed(clip_data)(data10, clip, box)
        data20 = dask.delayed(clip_data)(data20, clip, box)
        data10, data20 = dask.compute(data10, data20)
        data10, data20, data60 = persist(data10, data20, data60)

    validated_descriptions_all = [*dic_10m, *dic_20m, *dic_60m]

    # Super-resolving the 60m data into 10m bands
    sr60 = dsen2_60(data10, data20, data60, image_level)
    # Super-resolving the 20m data into 10m bands"
    sr20 = dsen2_20(data10, data20, image_level)

    sr_final = xarray.concat((data10.astype('uint16'), sr20.astype('uint16'), sr60.astype('uint16')), dim='band')

    sr_final.attrs['long_name'] = tuple(validated_descriptions_all)

    sr_final = persist(sr_final)
    
    # This code is needed if you want to write superresolution result to file
    #path_to_output_img = Path(input_dir).stem + "_superresolution.tif"
    #filename = os.path.join(input_dir, path_to_output_img)
    #sr_final.rio.to_raster(filename, compress='deflate', PREDICTOR=2, ZLEVEL=9, BIGTIFF='IF_SAFER', tiled=True, windowed=True, lock=True)
    gc.collect()
    return sr_final


def get_data(input_dir):
    """
    This function returns the raster data set of original image for
    all the available resolutions.
    """
    data_path = ""
    for file in glob.iglob(os.path.join(input_dir, "MTD*.xml"), recursive=True):
        data_path = file
    
    # The following line will define whether image is L1C or L2A
    # For instance image_level can be "MSIL1C" or "MSIL2A"
    image_level = Path(data_path).stem.split("_")[1]
    with rasterio.open(data_path) as rd:
        datasets = rd.subdatasets
    datasets = [
        next(x for x in datasets if ':10m:' in x),
        next(x for x in datasets if ':20m:' in x),
        next(x for x in datasets if ':60m:' in x)
    ]
    return datasets, image_level

    
def read_ds(dataset):
    """
    This function reads datasets and perform validation.
    """
    select_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    validated_indices = []
    validated_descriptions = []
    with rioxarray.open_rasterio(dataset, chunks=True, lock=True) as tif:
        bands = persist(tif)
        for i in range(bands.shape[0]):
            desc = validate_description(bands.long_name[i])
            name = get_band_short_name(desc)
            if name in select_bands:
                validated_descriptions.append(name)
                validated_indices.append(i)
        bands = bands.isel(band = validated_indices)
        assert bands.shape[1] >= 192 and bands.shape[2] >= 192
    return bands, validated_descriptions

    
def validate_description(description: str) -> str:
    """
    This function rewrites the description of each band in the given data set.

    Args:
        description: The actual description of a chosen band.

    Examples:
        >>> ds10.descriptions[0]
        'B4, central wavelength 665 nm'
        >>> validate_description(ds10.descriptions[0])
        'B4 (665 nm)'
    """
    m_re = re.match(r"(.*?), central wavelength (\d+) nm", description)
    if m_re:
        return m_re.group(1) + " (" + m_re.group(2) + " nm)"
    return description

    
def get_band_short_name(description: str) -> str:
    """
    This function returns only the name of the bands at a chosen resolution.

    Args:
        description: This is the output of the validate_description method.

    Examples:
        >>> desc = validate_description(ds10.descriptions[0])
        >>> desc
        'B4 (665 nm)'
        >>> get_band_short_name(desc)
        'B4'
    """
    if "," in description:
        return description[: description.find(",")]
    if " " in description:
        return description[: description.find(" ")]
    return description[:3]


def get_clip(data60, clip):
    """
    This function clips 60m bands and get bbox for clipping other bands.
    """
    shape = gpd.read_file(clip).to_crs(data60.rio.crs)
    shape = convert_3D_2D(shape)
    data60 = data60.rio.clip(shape.geometry.values, shape.crs, all_touched = True)
    box = list(data60.rio.bounds())
    return data60, box


def clip_data(data, clip, box):
    """
    This function clips 10 and 20m bands.
    """
    shape = gpd.read_file(clip).to_crs(data.rio.crs)
    shape = convert_3D_2D(shape)
    data = data.rio.clip_box(box[0], box[1], box[2], box[3])
    data = data.rio.clip(shape.geometry.values, shape.crs, all_touched=True, drop=False)
    return data