import numpy as np

import rioxarray


def normalize_file(input_file, output_file, minimum, maximum):
    # Read
    with rioxarray.open_rasterio(input_file, chunks = True, lock = True) as img:
        nodata = img.rio.nodata
        # Setting min and max to min and max of dtype
        if minimum == None or maximum == None:
            minimum = np.iinfo(img.dtype).min
            maximum = np.iinfo(img.dtype).max
        # Normalization
        img = (img - minimum) / (maximum - minimum)
        nodata = (nodata - minimum) / (maximum - minimum)
        img.rio.write_nodata(nodata, inplace = True)
        # Write
        img.rio.to_raster(output_file, compress = 'deflate', PREDICTOR = 2, ZLEVEL = 9, BIGTIFF = 'IF_SAFER', tiled = True, windowed = True, lock = True)
    
        