import numpy as np

import rasterio as rio


def normalize_file(input_file, output_file, minimum, maximum):
    #read
    with rio.open(input_file) as b:
        img = b.read()
        meta = b.profile
        transform = b.transform
        nodata = b.nodata
        crs = b.crs
    #setting min and max to min and max of dtype
    if minimum == None or maximum == None:
        minimum = np.iinfo(img.dtype).min
        maximum = np.iinfo(img.dtype).max
    #normalization
    img = (img - minimum) / (maximum - minimum)
    #write
    with rio.open(
        output_file,
        'w',
        driver='GTiff',
        height=img.shape[1],
        width=img.shape[2],
        count=img.shape[0],
        dtype=img.dtype,
        compress = 'deflate',
        PREDICTOR = 1,
        ZLEVEL=9,
        crs=crs,
        transform=transform,
        nodata = nodata
    ) as outfile:
        outfile.write(img)
    
        