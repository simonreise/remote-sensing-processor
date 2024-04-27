import numpy as np

import geopandas as gpd
import rioxarray
from geocube.api.core import make_geocube

from remote_sensing_processor.common.common_functions import convert_3D_2D, persist


def rasterize_vector(vector, raster, burn_value, output_file, nodata):
    with rioxarray.open_rasterio(raster, chunks=True, lock=True) as tif:
        # Read raster
        vector = gpd.read_file(vector)
        vector = convert_3D_2D(vector)
        assert np.issubdtype(vector[burn_value].dtype, np.number)
        rasterized = make_geocube(
            vector,
            measurements=[burn_value],
            like=tif,
            fill=nodata,
        )
        rasterized = persist(rasterized)
        rasterized = rasterized[burn_value].chunk('auto')
        rasterized = persist(rasterized)
        # Because predictor = 2 works with float64 only when libtiff > 3.2.0 is installed
        # and default libtiff in ubuntu is 3.2.0
        if rasterized.dtype == 'float64':
            rasterized = rasterized.astype('float32')
            rasterized = persist(rasterized)
    # Write
    rasterized.rio.to_raster(
        output_file,
        compress='deflate',
        PREDICTOR=2,
        ZLEVEL=9,
        BIGTIFF='IF_SAFER',
        tiled=True,
        NUM_THREADS='NUM_CPUS',
        lock=True,
    )
