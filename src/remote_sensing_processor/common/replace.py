import numpy as np

import rasterio as rio


def replace_val(input_file, output_file, new, old, nodata):
    #read
    with rio.open(input_file) as b:
        img = b.read()
        meta = b.profile
        transform = b.transform
        nodata_val = b.nodata
        crs = b.crs
    if old == None and nodata == True:
        old = nodata_val
    #replacing nodata value
    img = np.where(img == old, new, img)
    #write
    if nodata == True:
        nodata_val = new
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
        nodata = nodata_val
    ) as outfile:
        outfile.write(img)