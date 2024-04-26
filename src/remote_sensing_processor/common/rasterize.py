import numpy as np

import geopandas as gpd
import rasterio as rio
from rasterio import features

from remote_sensing_processor.common.common_functions import convert_3D_2D


def rasterize_vector(vector, raster, burn_value, output_file, nodata):
    # Read raster
    with rio.open(raster) as b:
        img = b.read()
        meta = b.profile
        shape = b.shape
        transform = b.transform
        crs = b.crs
    # Read vector file
    shapes = gpd.read_file(vector).to_crs(crs)
    geoms = convert_3D_2D(shapes)
    geom_value = ((geom,value) for geom, value in zip(geoms, shapes[burn_value]))
    if len(shape) > 2:
        shape = shape[-2:]
    rasterized = features.rasterize(geom_value,
                                    out_shape=shape,
                                    transform=transform,
                                    fill=nodata,
                                    dtype=shapes[burn_value].dtype)
    # Write
    with rio.open(
        output_file,
        'w',
        driver='GTiff',
        height=img.shape[1],
        width=img.shape[2],
        count=img.shape[0],
        dtype=shapes[burn_value].dtype,
        compress='deflate',
        PREDICTOR=2,
        ZLEVEL=9,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as outfile:
        outfile.write(rasterized, 1)