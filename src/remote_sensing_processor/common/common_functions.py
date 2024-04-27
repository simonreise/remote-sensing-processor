import warnings

import rasterio as rio
import shapely
from rasterio.enums import Resampling


def convert_3D_2D(df):
    df.geometry = shapely.force_2d(df.geometry)
    return df

 
def get_resampling(resample):
    if resample == 'bilinear':
        resample = Resampling.bilinear
    elif resample == 'cubic':
        resample = Resampling.cubic
    elif resample == 'cubic_spline':
        resample = Resampling.cubic_spline
    elif resample == 'lanczos':
        resample = Resampling.lanczos
    elif resample == 'average':
        resample = Resampling.average
    elif resample == 'mode':
        resample = Resampling.mode
    elif resample == 'max':
        resample = Resampling.max
    elif resample == 'min':
        resample = Resampling.min
    elif resample == 'med':
        resample = Resampling.med
    elif resample == 'q1':
        resample = Resampling.q1
    elif resample == 'q3':
        resample = Resampling.q3
    elif resample == 'sum':
        resample = Resampling.sum
    elif resample == 'rms':
        resample = Resampling.rms
    elif resample == 'nearest':
        resample = Resampling.nearest
    else:
        resample = Resampling.nearest
    return resample


def get_first_proj(img):
    with rio.open(img) as im:
        projection = im.crs
    return projection
    
    
def persist(*inputs):
    """
    This function tries to persist array if it is not too big to fit in memory.
    """
    enough_memory = True
    results = []
    # Trying to persist dataset in memory (it makes processing much faster)
    for i in inputs:
        if enough_memory == True:
            try:
                results.append(i.persist())
            except:
                warnings.warn("Dataset does not fit in memory. Processing can be much slower.")
                enough_memory = False
                results = inputs
                break
        else:
            results = inputs
    # Return array instead of tuple if it consists of one element
    results = tuple(results)
    if len(results) == 1:
        results = results[0]
    return results