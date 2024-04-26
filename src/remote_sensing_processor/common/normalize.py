import numpy as np

import rioxarray

from remote_sensing_processor.common.common_functions import persist


def normalize_file(input_file, output_file, minimum, maximum):
    # Read
    with rioxarray.open_rasterio(input_file, chunks=True, lock=True) as img:
        nodata = img.rio.nodata
        # Setting min and max to min and max of dtype
        if minimum == None or maximum == None:
            minimum = np.iinfo(img.dtype).min
            maximum = np.iinfo(img.dtype).max
        # Normalization
        img = (img - minimum) / (maximum - minimum)
        nodata = (nodata - minimum) / (maximum - minimum)
        img.rio.write_nodata(nodata, inplace=True)
        # Because predictor = 2 works with float64 only when libtiff > 3.2.0 is installed
        # and default libtiff in ubuntu is 3.2.0
        if img.dtype == 'float64':
            img = img.astype('float32')
        img = persist(img)
    # Write
    img.rio.to_raster(
        output_file, 
        compress='deflate', 
        PREDICTOR=2, 
        ZLEVEL=9, 
        BIGTIFF='IF_SAFER', 
        tiled=True, 
        NUM_THREADS='NUM_CPUS', 
        lock=True,
    )
    
        