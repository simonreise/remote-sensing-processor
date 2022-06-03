import numpy as np

import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon, shape, Point
from rasterio.enums import Resampling


def convert_3D_2D(df):
    '''
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    '''
    new_geo = []
    for p in df.geometry:
        try:
            if p.has_z:
                if p.geom_type == 'Polygon':
                    lines = [xy[:2] for xy in list(p.exterior.coords)]
                    new_p = Polygon(lines)
                    new_geo.append(new_p)
                elif p.geom_type == 'MultiPolygon':
                    new_multi_p = []
                    for ap in p:
                        lines = [xy[:2] for xy in list(ap.exterior.coords)]
                        new_p = Polygon(lines)
                        new_multi_p.append(new_p)
                    new_geo.append(MultiPolygon(new_multi_p))
            else:
                new_geo.append(p)
        except:
            new_geo.append(p)
    
    return new_geo
    
def get_resampling(resample):
    if resample == 'bilinear':
        resample = Resampling.bilinear
    elif resample == 'cubic':
        resample == Resampling.cubic
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
    else:
        resample = Resampling.nearest
    return resample