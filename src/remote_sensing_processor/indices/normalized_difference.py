import os
import sys
import warnings

import numpy as np

import rioxarray


warnings.filterwarnings("ignore", message = "divide by zero")
warnings.filterwarnings("ignore", message = "invalid value encountered")

def nd(name, b1, b2, folder = None):
    with rioxarray.open_rasterio(b1, chunks = True, lock = True) as band1:
        with rioxarray.open_rasterio(b2, chunks = True, lock = True) as band2:
            if band1.shape != band2.shape:
                raise RuntimeError('Datasets have different shapes')
            elif band1.rio.transform() != band2.rio.transform():
                raise RuntimeError('Datasets have different transforms')
            else:
                with np.errstate(divide='ignore', invalid = 'ignore'):
                    final = (band1.astype('float64') - band2.astype('float64')) / (band1.astype('float64') + band2.astype('float64'))
                final = final.fillna(0)
                if folder == None:
                    savefolder = os.path.dirname(b1)
                else:
                    savefolder = folder
            # Because predictor = 2 works with float64 only when libtiff > 3.2.0 is installed and default libtiff in ubuntu is 3.2.0
            if final.dtype == 'float64':
                final = final.astype('float32')
            # Write
            final.rio.write_nodata(0, inplace = True)
            final.rio.to_raster(savefolder + '/' + name + '.tif', compress = 'deflate', PREDICTOR = 2, ZLEVEL = 9, BIGTIFF='IF_SAFER', tiled = True, NUM_THREADS = 'NUM_CPUS', lock = True)
    return savefolder + '/' + name + '.tif'