import numpy as np

import rioxarray

from remote_sensing_processor.common.common_functions import persist


def replace_val(input_file, output_file, new, old, nodata):
    # Read
    with rioxarray.open_rasterio(input_file, chunks=True, lock=True) as tif:
        img = tif
        if old == None and nodata == True:
            old = img.rio.nodata
        # Replacing nodata value
        img = img.where(img != old, new)
        # Because predictor = 2 works with float64 only when libtiff > 3.2.0 is installed
        # and default libtiff in ubuntu is 3.2.0
        if img.dtype == 'float64':
            img = img.astype('float32')
        # Rewrite nodata
        if nodata == True:
            img.rio.write_nodata(new, inplace=True)
        img = persist(img)
    # Write
    img.rio.to_raster(
        output_file, 
        compress='deflate', 
        PREDICTOR=2, 
        ZLEVEL=9, 
        BIGTIFF='IF_SAFER', 
        tiled=True, 
        NUM_THREADS='ALL_CPUS',
        lock=True,
    )