import numpy as np
from glob import glob
import os
import warnings
import re

import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon, shape, Point
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.merge
import rasterio.fill
import rasterio.mask
from rasterio.io import MemoryFile
from rasterio.enums import Resampling

from remote_sensing_processor.common.common_functions import convert_3D_2D

from remote_sensing_processor.imagery_types.types import get_type


def mosaic_main(inputs, output_dir, fill_nodata, fill_distance, clipper, crs, nodata, reference_raster, mb, keep_all_channels):
    if reference_raster != None:
        with rio.open(reference_raster) as r:
            crs = r.crs
    if mb == True:
        bands = get_bands(inputs, keep_all_channels)
        #print(bands)
        for b in bands:
            #opening files
            band = b['name']
            files = []
            for path in b['bands']:
                pathfile = rio.open(path)
                pathfile, crs = check_crs(pathfile = pathfile, crs = crs, nodata = nodata)
                files.append(pathfile)
            mosaic_process(files = files, output_dir = output_dir, fill_nodata = fill_nodata, fill_distance = fill_distance, clipper = clipper, crs = crs, nodata = nodata, reference_raster = reference_raster, band = band)
            print('Processing band ' + band + ' is completed')
    else:
        files = []
        for inp in inputs:
            pathfile = rio.open(inp)
            pathfile, crs = check_crs(pathfile = pathfile, crs = crs,  nodata = nodata)
            files.append(pathfile)
        band = os.path.basename(inputs[0])[:-4]+'_mosaic'
        mosaic_process(files = files, output_dir = output_dir, fill_nodata = fill_nodata, fill_distance = fill_distance, clipper = clipper, crs = crs, nodata = nodata, reference_raster = reference_raster, band = band)
        print('Processing completed')


def check_crs(pathfile, crs, nodata):
    if crs == None:
        crs = pathfile.crs
        return pathfile, crs
    elif pathfile.crs != crs:
        #warnings.warn('File ' + pathfile.files[0] + ' have CRS ' + str(pathfile.crs) + ' which is different from ' + str(crs) + '. Reprojecting can be memory consuming. It is recommended to reproject all files to the same CRS before mosaicing.')
        orig = pathfile.read()
        orig_meta = pathfile.profile
        bounds = pathfile.bounds
        #reprojecting
        transform, width, height = calculate_default_transform(
            pathfile.crs, crs, orig_meta['width'], orig_meta['height'], *bounds)
        img = np.zeros((orig.shape[0], height, width), orig.dtype)
        reproject(
            source=orig,
            destination=img,
            src_transform=orig_meta['transform'],
            src_crs=orig_meta['crs'],
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest)
        memfile = MemoryFile()
        rst = memfile.open(
            driver='GTiff',
            height=img.shape[1],
            width=img.shape[2],
            count=img.shape[0],
            dtype=img.dtype,
            compress = 'lzw',
            crs=crs,
            transform=transform,
            nodata = nodata,
            BIGTIFF='YES')
        rst.write(img)
        return rst, crs
    else:
        return pathfile, crs


def mosaic_process(files, output_dir, fill_nodata, fill_distance, clipper, crs, nodata, reference_raster, band):
    #merging files
    final, final_trans = rio.merge.merge(files, method = 'last', nodata = nodata)
    #filling nodata
    if fill_nodata == True:
        mask = np.where(final == nodata, 0, 1)
        final = rio.fill.fillnodata(final, mask, max_search_distance=fill_distance)
    files = None
    memfile = None
    temp = None
    #clipping mosaic with vector mask
    if clipper != None:
        shape = gpd.read_file(clipper).to_crs(crs)
        shape = convert_3D_2D(shape)
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=final.shape[1],
                width=final.shape[2],
                count=final.shape[0],
                dtype=final.dtype,
                compress = 'lzw',
                crs=crs,
                transform=final_trans,
                BIGTIFF='YES',
                nodata = nodata
            ) as temp:
                temp.write(final)
                final, final_trans = rio.mask.mask(temp, shape, crop=True, filled=True)
    #resampling to the same shape and resolution as another raster
    if reference_raster != None:
        ref = rio.open(reference_raster)
        f1 = np.zeros((ref.shape), final.dtype)
        reproject(
            source = final,
            destination = f1,
            src_transform = final_trans,
            src_crs=crs,
            src_nodata = nodata,
            dst_transform = ref.transform,
            dst_resolution = ref.res,
            dst_crs=ref.crs,
            dst_nodata = nodata,
            num_threads = 4,
            resampling=Resampling.average)
        final = f1
        final_trans = ref.transform
        crs = ref.crs
    if final.ndim == 2:
        final = final[np.newaxis,:,:]
    with rio.open(
        output_dir + band + '.tif',
        'w',
        driver='GTiff',
        height=final.shape[1],
        width=final.shape[2],
        count=1,
        dtype=final.dtype,
        compress = 'lzw',
        crs=crs,
        transform=final_trans,
        BIGTIFF='YES',
        nodata = nodata
    ) as outfile:
        outfile.write(final)


def get_bands(paths, keep_all_channels):
    sets = []
    for path in paths:
        im_type = get_type(path)
        if im_type == 'Sentinel2_up':
            path = path + 'GRANULE\\'
            path = glob(path + '*')[0]
            path = path + '\\IMG_DATA\\'
            if os.path.isdir(path + 'R10m\\'):
                bands = glob(path + 'R10m\\*B*.jp2') + glob(path + 'R20m\\*B*.jp2') + glob(path + 'R60m\\*B*.jp2')
            else:
                bands = glob(path + '*B*.jp2')
        else:
            bands = glob(path + '*.*')
        sets.append([bands, im_type])
    # getting imagery type and bands list
    unique_types = set(x[1] for x in sets)
    if (unique_types.issubset(['Landsat8_up_l1', 'Landsat7_up_l1', 'Landsat5_up_l1', 'Landsat1_up_l1'])) or (unique_types.issubset(['Landsat8_up_l2', 'Landsat7_up_l2', 'Landsat5_up_l2', 'Landsat1_up_l2'])) or (unique_types.issubset(['Landsat8_p', 'Landsat7_p', 'Landsat5_p', 'Landsat1_p'])):
        b1 = {'name': 'B1', 'bands': []} #coastal/aerosol
        b2 = {'name': 'B2', 'bands': []} #blue
        b3 = {'name': 'B3', 'bands': []} #green
        b4 = {'name': 'B4', 'bands': []} #red
        b5 = {'name': 'B5', 'bands': []} #nir1
        b6 = {'name': 'B6', 'bands': []} #swir1
        b7 = {'name': 'B7', 'bands': []} #swir2
        b8 = {'name': 'B8', 'bands': []} #pan
        b9 = {'name': 'B9', 'bands': []} #cirrus
        b10 = {'name': 'B10', 'bands': []} #t1
        b11 = {'name': 'B11', 'bands': []} #t2
        for s in sets:
            bands = s[0]
            lsver = s[1]
            bands = [band for band in bands if '_GM_' not in band]
            for band in bands:
                if ('B1.' in band) or ('B01' in band):
                    if 'Landsat8' in lsver:
                        b1['bands'].append(band)
                    elif 'Landsat7' in lsver:
                        b2['bands'].append(band)
                    elif 'Landsat5' in lsver:
                        b2['bands'].append(band)
                    elif 'Landsat1' in lsver:
                        b3['bands'].append(band)
                elif ('B2' in band) or ('B02' in band):
                    if 'Landsat8' in lsver:
                        b2['bands'].append(band)
                    elif 'Landsat7' in lsver:
                        b3['bands'].append(band)
                    elif 'Landsat5' in lsver:
                        b3['bands'].append(band)
                    elif 'Landsat1' in lsver:
                        b4['bands'].append(band)
                elif ('B3' in band) or ('B03' in band):
                    if 'Landsat8' in lsver:
                        b3['bands'].append(band)
                    elif 'Landsat7' in lsver:
                        b4['bands'].append(band)
                    elif 'Landsat5' in lsver:
                        b4['bands'].append(band)
                    elif 'Landsat1' in lsver:
                        b5['bands'].append(band)
                elif ('B4' in band) or ('B04' in band):
                    if 'Landsat8' in lsver:
                        b4['bands'].append(band)
                    elif 'Landsat7' in lsver:
                        b5['bands'].append(band)
                    elif 'Landsat5' in lsver:
                        b5['bands'].append(band)
                    elif 'Landsat1' in lsver:
                        b5['bands'].append(band)
                elif ('B5' in band) or ('B05' in band):
                    if 'Landsat8' in lsver:
                        b5['bands'].append(band)
                    elif 'Landsat7' in lsver:
                        b6['bands'].append(band)
                    elif 'Landsat5' in lsver:
                        b6['bands'].append(band)
                elif ('B6' in band) or ('B06' in band):
                    if 'Landsat8' in lsver:
                        b6['bands'].append(band)
                    elif 'Landsat7' in lsver:
                        if 'VCID_1' in band:
                            b10['bands'].append(band)
                        elif 'VCID_2' in band:
                            b11['bands'].append(band)
                    elif 'Landsat5' in lsver:
                        b10['bands'].append(band)
                elif ('B7' in band) or ('B07' in band):
                    if 'Landsat8' in lsver:
                        b7['bands'].append(band)
                    elif 'Landsat7' in lsver:
                        b7['bands'].append(band)
                    elif 'Landsat5' in lsver:
                        b7['bands'].append(band)
                elif ('B8' in band) or ('B08' in band):
                    if 'Landsat8' in lsver:
                        b8['bands'].append(band)
                    elif 'Landsat7' in lsver:
                        b8['bands'].append(band)
                elif ('B9' in band) or ('B09' in band):
                    if 'Landsat8' in lsver:
                        b9['bands'].append(band)
                elif ('B10' in band):
                    if 'Landsat8' in lsver:
                        b10['bands'].append(band)
                elif ('B11' in band):
                    if 'Landsat8' in lsver:
                        b11['bands'].append(band)
        bands = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11]
    elif unique_types.issubset(['Sentinel2_up', 'Sentinel2_p']):
        b1 = {'name': 'B1', 'bands': []}
        b2 = {'name': 'B2', 'bands': []}
        b3 = {'name': 'B3', 'bands': []}
        b4 = {'name': 'B4', 'bands': []}
        b5 = {'name': 'B5', 'bands': []}
        b6 = {'name': 'B6', 'bands': []}
        b7 = {'name': 'B7', 'bands': []}
        b8 = {'name': 'B8', 'bands': []}
        b8a = {'name': 'B8A', 'bands': []}
        b9 = {'name': 'B9', 'bands': []}
        b10 = {'name': 'B10', 'bands': []}
        b11 = {'name': 'B11', 'bands': []}
        b12 = {'name': 'B12', 'bands': []}
        for s in sets:
            bands = s[0]
            sver = s[1]
            for band in bands:
                if ('B01' in band) or ('B01_60m' in band) or ('B1.tif' in band):
                    b1['bands'].append(band)
                elif ('B02' in band) or ('B02_10m' in band) or ('B2.tif' in band):
                    b2['bands'].append(band)
                elif ('B03' in band) or ('B03_10m' in band) or ('B3.tif' in band):
                    b3['bands'].append(band)
                elif ('B04' in band) or ('B04_10m' in band) or ('B4.tif' in band):
                    b4['bands'].append(band)
                elif ('B05' in band) or ('B05_20m' in band) or ('B5.tif' in band):
                    b5['bands'].append(band)
                elif ('B06' in band) or ('B06_20m' in band) or ('B6.tif' in band):
                    b6['bands'].append(band)
                elif ('B07' in band) or ('B07_20m' in band) or ('B7.tif' in band):
                    b7['bands'].append(band)
                elif ('B08' in band) or ('B08_10m' in band) or ('B8.tif' in band):
                    b8['bands'].append(band)
                elif ('B8A' in band) or ('B8A_20m' in band) or ('B8A.tif' in band):
                    b8a['bands'].append(band)
                elif ('B09' in band) or ('B09_60m' in band) or ('B9.tif' in band):
                    b9['bands'].append(band)
                elif ('B10' in band) or ('B10_60m' in band) or ('B10.tif' in band):
                    b10['bands'].append(band)
                elif ('B11' in band) or ('B11_20m' in band) or ('B11.tif' in band):
                    b11['bands'].append(band)
                elif ('B12' in band) or ('B12_20m' in band) or ('B12.tif' in band):
                    b12['bands'].append(band)
        bands = [b1, b2, b3, b4, b5, b6, b7, b8, b8a, b9, b10, b11, b12]
    else:
        allbands = []
        for s in sets:
            bands = s[0]
            for band in bands:
                band = os.path.basename(band)
                allbands.append(band)
        allbands = list(set(bands))
        final_bands = []
        for bandname in allbands:
            thisband = {'name': bandname, 'bands': []}
            for s in sets:
                bands = s[0]
                for band in bands:
                    if bandname in band:
                        thisband['bands'].append(band)
            final_bands.append(thisband)
        bands = final_bands
    final_bands = []
    for i in bands:
        if (len(i['bands']) != 0) and (((len(i['bands']) == len(paths)) and (keep_all_channels == False)) or (keep_all_channels == True)):
            final_bands.append(i)
    return final_bands
    

def order(dirs):
    zeros = []
    for path in dirs:
        path = glob(path + '*.tif')[0]
        with rio.open(path) as bnd:
            img = bnd.read(1)
        try:
            zeros.append(np.count_nonzero(img==0))
        except:
            zeros.append(0)
    zerosdict = dict(zip(dirs, zeros))
    sortedzeros = dict(sorted(zerosdict.items(), key=lambda item: item[1], reverse = True))
    order = list(sortedzeros.keys())
    return order
    
    
def ismultiband(inp):
    return os.path.isdir(inp)