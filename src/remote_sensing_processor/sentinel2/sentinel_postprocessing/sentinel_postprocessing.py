import numpy as np
from glob import glob

import xarray
import dask

import geopandas as gpd
import rasterio as rio
import rasterio.fill
from rasterio.enums import Resampling
import rioxarray

from remote_sensing_processor.common.common_functions import convert_3D_2D, get_resampling, persist


def s2postprocess_superres(
    img, 
    projection, 
    cloud_mask, 
    clip, 
    normalize, 
    path, 
    path1,
):
    bandnames = img.long_name
    outfiles = s2postprocess(
        img=img, 
        projection=projection, 
        cloud_mask=cloud_mask, 
        clip=clip, 
        normalize=normalize, 
        path=path, 
        path1=path1, 
        bandoutnames=bandnames,
    )
            
            
def s2postprocess_no_superres(
    projection, 
    cloud_mask, 
    clip, 
    normalize, 
    resample, 
    path, 
    path1, 
    upscale,
):
    if upscale == 'resample':
        resample = get_resampling(resample)
    # Getting bands with different resolution
    if 'MSIL2A' in path1:
        bands10 = []
        bands20 = []
        bands60 = []
        bnames10 = []
        bnames20 = []
        bnames60 = []
        bandnames = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        bandoutnames = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        for i in range(len(bandnames)):
            band = glob(path1 + '/**/*' + bandnames[i] + '_10m.jp2', recursive=True) 
            if band != []:
                bands10.append(band[0])
                bnames10.append(bandoutnames[i])
            else:
                band = glob(path1 + '/**/*' + bandnames[i] + '_20m.jp2', recursive=True)
                if band != []:
                    bands20.append(band[0])
                    bnames20.append(bandoutnames[i])
                else: 
                    band = glob(path1 + '/**/*' + bandnames[i] + '_60m.jp2', recursive=True)
                    if band != []:
                        bands60.append(band[0])
                        bnames60.append(bandoutnames[i])
    else:
        bands10 = ['B02', 'B03', 'B04', 'B08']
        bands20 = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        bands60 = ['B01', 'B09', 'B10']
        bnames10 = ['B2', 'B3', 'B4', 'B8']
        bnames20 = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
        bnames60 = ['B1', 'B9', 'B10']
        for i in range(len(bands10)):
            band = glob(path1 + '/**/*' + bands10[i] + '.jp2', recursive=True)
            bands10[i] = band[0]
        for i in range(len(bands20)):
            band = glob(path1 + '/**/*' + bands20[i] + '.jp2', recursive=True)
            bands20[i] = band[0]
        for i in range(len(bands60)):
            band = glob(path1 + '/**/*' + bands60[i] + '.jp2', recursive=True)
            bands60[i] = band[0]
    files = []
    outfiles = []
    # Processing 10 m bands
    for i in bands10:
        with rioxarray.open_rasterio(i, chunks=True, lock=True) as tif:
            files.append(persist(tif))
    # If no upscaling needed then process different resolution bands separately
    if upscale == None:
        bandoutnames = bnames10
        img = xarray.concat(files, dim=xarray.Variable('band', bandoutnames))
        outfiles.extend(s2postprocess(
            img=img, 
            projection=projection, 
            cloud_mask=cloud_mask, 
            clip=clip, 
            normalize=normalize, 
            path=path, 
            path1=path1, 
            bandoutnames=bandoutnames,
        ))
        files = []
    # Processing 20 m bands
    for i in bands20:
        with rioxarray.open_rasterio(i, chunks=True, lock=True) as tif:
            band = persist(tif)
            if upscale == 'resample':
                band = band.rio.reproject_match(files[0], resampling=resample)
            files.append(band)
    # If no upscaling needed then process different resolution bands separately
    if upscale == None:
        bandoutnames = bnames20
        img = xarray.concat(files, dim=xarray.Variable('band', bandoutnames))
        outfiles.extend(s2postprocess(
            img=img, 
            projection=projection, 
            cloud_mask=cloud_mask, 
            clip=clip, 
            normalize=normalize, 
            path=path, 
            path1=path1, 
            bandoutnames=bandoutnames,
        ))
        files = []
    # Processing 60 m bands
    for i in bands60:
        with rioxarray.open_rasterio(i, chunks=True, lock=True) as tif:
            band = persist(tif)
            if upscale == 'resample':
                band = band.rio.reproject_match(files[0], resampling=resample)
            files.append(band)
    # If no upscaling needed then process different resolution bands separately
    if upscale == None:
        bandoutnames = bnames60
        img = xarray.concat(files, dim=xarray.Variable('band', bandoutnames))
        outfiles.extend(s2postprocess(
            img=img, 
            projection=projection, 
            cloud_mask=cloud_mask, 
            clip=clip, 
            normalize=normalize, 
            path=path, 
            path1=path1, 
            bandoutnames=bandoutnames,
        ))
    # If upscaling needed then process all bands together
    if upscale == 'resample':
        bandoutnames = bnames10 + bnames20 + bnames60
        img = xarray.concat(files, dim = xarray.Variable('band', bandoutnames))        
        outfiles.extend(s2postprocess(
            img=img, 
            projection=projection, 
            cloud_mask=cloud_mask, 
            clip=clip, 
            normalize=normalize, 
            path=path, 
            path1=path1, 
            bandoutnames=bandoutnames,
        ))


def s2postprocess(
    img, 
    projection, 
    cloud_mask, 
    clip, 
    normalize, 
    path, 
    path1, 
    bandoutnames
):
    try:
        if img.rio.nodata == None:
            nodata = 0
        else:
            nodata = img.rio.nodata
    except:
        nodata = 0
    img.rio.write_nodata(nodata, inplace=True)
    img = persist(img)
    # Masking 65555 pixels
    img = img.where(img <= 10000, 1)
    img = xarray.apply_ufunc(
        rio.fill.fillnodata, 
        img, 
        xarray.where(img == 1, 0, 1), 
        dask='parallelized', 
        keep_attrs='override', 
        kwargs={'max_search_distance': 500},
    )
    img = img.chunk('auto')
    img = persist(img)
    if cloud_mask == True:
        # Masking clouds by mask
        shape = glob(path1 + '/GRANULE/**/QI_DATA/MSK_CLOUDS_B00.gml')
        if shape != []:
            cloudmask = gpd.read_file(shape[0]).to_crs(img.rio.crs)
            cloudmask = convert_3D_2D(cloudmask)
            img = img.rio.clip(cloudmask.geometry.values, cloudmask.crs, invert=True)
        # Masking clouds by scl mask and cloud proba
        sclpath = glob(path1 + r'/GRANULE/*/IMG_DATA/R20m/*SCL_20m.jp2')
        if sclpath != []:
            with rioxarray.open_rasterio(sclpath[0], chunks=True, lock=True) as tif:
                mask = persist(tif)
            mask = mask.rio.reproject_match(img, resampling=Resampling.nearest)
            mask = xarray.where(mask.isin([0, 1, 3, 8, 9, 10]), 0, 1) # 0 - nodata, 1 - data
        else:
            mask = None
        clmpath = glob(path1 + r'/GRANULE/**/QI_DATA/MSK_CLDPRB_20m.jp2')
        if clmpath != []:
            with rioxarray.open_rasterio(clmpath[0], chunks = True, lock = True) as tif:
                clmask = persist(tif)
            clmask = clmask.rio.reproject_match(img, resampling = Resampling.nearest)
            if not isinstance(mask, type(None)):
                mask = mask.where(clmask < 10, 0) # 0 - nodata, 1 - data
            else:
                mask = xarray.where(clmask >= 10, 0, 1) # 0 - nodata, 1 - data
        if not isinstance(mask, type(None)):
            # Delete small gaps in mask
            mask = xarray.where((xarray.apply_ufunc(
                rio.fill.fillnodata, 
                mask, 
                mask, 
                dask='parallelized', 
                keep_attrs='override', 
                kwargs={'max_search_distance': 5},
            )) == 1, 0, 1) # fill 0 - nodata -> 1 - nodata, 0 - data
            # Emerge mask
            mask = xarray.apply_ufunc(
                rio.fill.fillnodata, 
                mask, 
                mask, 
                dask='parallelized', 
                keep_attrs='override', 
                kwargs={'max_search_distance': 22},
            ) # fill 0 - data
            # Masking nodata
            img = img.where(mask.squeeze() == 0, 0) # 0 - data, 1 - nodata
        img = persist(img)
    # Reprojection
    if projection != None:
        img = img.rio.reproject(projection, resampling=Resampling.nearest)
        img = img.chunk('auto')
        img = persist(img)
    else:
        projection = img.rio.crs
    # Clipping
    if clip != None:
        shape = gpd.read_file(clip).to_crs(projection)
        shape = convert_3D_2D(shape)
        img = img.rio.clip(shape.geometry.values, shape.crs)
        img = persist(img)
    # Normalization
    if normalize == True:
        img = img / 10000
        # Because predictor = 2 works with float64 only when libtiff > 3.2.0 is installed
        # and default libtiff in ubuntu is 3.2.0
        if img.dtype == 'float64':
            img = img.astype('float32')
        img = persist(img)
    # Save
    if 'long_name' in img.attrs:
        del img.attrs['long_name']
    outfiles = []
    results = []
    for i in range(img.shape[0]):
        pathres = path + '/' + bandoutnames[i] + '.tif'
        outfiles.append(pathres)
        results.append(img[i].rio.to_raster(
            pathres, 
            compress='deflate', 
            PREDICTOR=2, 
            ZLEVEL=9, 
            BIGTIFF='IF_SAFER', 
            tiled=True, 
            NUM_THREADS='NUM_CPUS', 
            compute=False, 
            lock=True,
        ))
    dask.compute(*results)
    return outfiles