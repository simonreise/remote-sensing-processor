import numpy as np
from glob import glob
import os


import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon, shape, Point
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.merge
import rasterio.fill
import rasterio.mask
from rasterio.io import MemoryFile
from rasterio.enums import Resampling


def nd(name, b1, b2, folder = None):
    with rio.open(b1) as file1:
        band1 = file1.read()
        meta1 = file1.profile
    with rio.open(b2) as file2:
        band2 = file2.read()
        meta2 = file2.profile
    try:
        if band1.shape != band2.shape:
            raise RuntimeError('Datasets have different shapes')
        elif meta1 != meta2:
            raise RuntimeError('Datasets have different transforms')
        else:
            final = (band1.astype('float64') - band2.astype('float64')) / (band1.astype('float64') + band2.astype('float64'))
            if folder == None:
                savefolder = os.path.dirname(b1)
            else:
                savefolder = folder
            with rio.open(
                savefolder + '\\' + name + '.tif',
                'w',
                driver='GTiff',
                height=final.shape[1],
                width=final.shape[2],
                count=1,
                dtype=final.dtype,
                compress = 'lzw',
                crs=meta1['crs'],
                transform=meta1['transform'],
                BIGTIFF='YES',
                nodata = meta1['nodata']
            ) as outfile:
                outfile.write(final)
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    return savefolder + '\\' + name + '.tif'