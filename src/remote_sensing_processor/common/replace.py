import numpy as np

import rioxarray


def replace_val(input_file, output_file, new, old, nodata):
    # Read
    with rioxarray.open_rasterio(input_file, chunks = True, lock = True) as tif:
        img = tif
        if old == None and nodata == True:
            old = img.rio.nodata
        # Replacing nodata value
        img = img.where(img != old, new)
        # Write
        if nodata == True:
            img.rio.write_nodata(new, inplace=True)
        img.rio.to_raster(output_file, compress = 'deflate', PREDICTOR = 2, ZLEVEL = 9, BIGTIFF = 'IF_SAFER', tiled = True, windowed = True, lock = True)