from glob import glob
import os
import warnings

import numpy as np
from skimage.exposure import match_histograms
import dask
import xarray

import geopandas as gpd
import rasterio as rio
import rasterio.fill
import rioxarray
from rioxarray.merge import merge_arrays

from remote_sensing_processor.common.common_functions import convert_3D_2D, get_resampling, persist

from remote_sensing_processor.imagery_types.types import get_type


def mosaic_main(inputs, output_dir, fill_nodata, fill_distance, clip, crs, nodata, reference_raster, resample, match_hist, mb, keep_all_channels):
    paths = []
    resample = get_resampling(resample)
    if reference_raster != None:
        with rio.open(reference_raster) as r:
            crs = r.crs
    if mb == True:
        bands = get_bands(inputs, keep_all_channels)
        for b in bands:
            # Opening files
            band = b['name']
            path = proc_files(inputs = b['bands'], output_dir = output_dir, fill_nodata = fill_nodata, fill_distance = fill_distance, clip = clip, crs = crs, nodata = nodata, reference_raster = reference_raster, resample = resample, band = band, match_hist = match_hist)
            paths.append(path)
            print('Processing band ' + band + ' is completed')
    else:
        band = os.path.basename(inputs[0])[:-4]+'_mosaic'
        path = proc_files(inputs = inputs, output_dir = output_dir, fill_nodata = fill_nodata, fill_distance = fill_distance, clip = clip, crs = crs, nodata = nodata, reference_raster = reference_raster, resample = resample, band = band, match_hist = match_hist)
        paths.append(path)
        print('Processing completed')
    return paths
    
    
def proc_files(inputs, output_dir, fill_nodata, fill_distance, clip, crs, nodata, reference_raster, resample, band, match_hist):
    files = []
    ref_hist = None
    futures = []
    for inp in inputs:
        # If histogram matching is needed then we need the histogram of the first file to process other files, so we cannot do it in parallel
        if match_hist and isinstance(ref_hist, type(None)):
            first, ref_hist = prepare_file(inp = inp, crs = crs, nodata = nodata, clip = clip, match_hist = match_hist, ref_hist = ref_hist)
        else:
            futures.append(dask.delayed(prepare_file)(inp = inp, crs = crs, nodata = nodata, clip = clip, match_hist = match_hist, ref_hist = ref_hist))
    files = dask.compute(*futures)
    files = list(files)
    # Adding first file
    if match_hist:
        files.insert(0, first)
    path = mosaic_process(files = files, output_dir = output_dir, fill_nodata = fill_nodata, fill_distance = fill_distance, clip = clip, crs = crs, nodata = nodata, reference_raster = reference_raster, resample = resample, band = band)
    return path


def prepare_file(inp, crs, nodata, clip, match_hist, ref_hist):
    with rioxarray.open_rasterio(inp, chunks = True, lock = True) as tif:
        pathfile = persist(tif)
    # If nodata not defined then read nodata from first file or set to 0
    if nodata == None:
        if pathfile.rio.nodata == None:
            nodata = 0
        else:
            nodata = pathfile.rio.nodata
    pathfile.rio.write_nodata(nodata, inplace = True)
    if crs == None:
        crs = pathfile.rio.crs
    if pathfile.rio.crs != crs:
        #warnings.warn('File ' + pathfile.files[0] + ' have CRS ' + str(pathfile.crs) + ' which is different from ' + str(crs) + '. Reproject can be memory consuming. It is recommended to reproject all files to the same CRS before mosaicing.')
        pathfile = pathfile.rio.reproject(crs)
        pathfile = persist(pathfile)
    if clip != None:
        shape = gpd.read_file(clip).to_crs(crs)
        shape = convert_3D_2D(shape)
        pathfile = pathfile.rio.clip(shape)
        pathfile = persist(pathfile)
    # Reading histogram if it is the first file
    if match_hist and isinstance(ref_hist, type(None)):
        mean = pathfile.where(pathfile != nodata).mean().item()
        ref_hist = pathfile.where(pathfile != nodata, mean)
        pathfile = persist(pathfile)
        return pathfile, ref_hist
    # Histogram matching
    elif match_hist:
        mean = pathfile.where(pathfile != nodata).mean().item()
        filled = pathfile.where(pathfile != nodata, mean)
        matched = match_histograms(filled.data, ref_hist.data)
        pathfile = pathfile.where(pathfile == nodata, matched)
        pathfile = persist(pathfile)
    return pathfile


def mosaic_process(files, output_dir, fill_nodata, fill_distance, clip, crs, nodata, reference_raster, resample, band):
    # Nodata check
    if nodata == None:
        nodata = files[0].rio.nodata
    for file in files:
        assert file.rio.nodata == nodata
    # Merging files
    final = merge_arrays(files, method = 'first', nodata = nodata)
    final = persist(final)
    # Filling nodata
    if fill_nodata == True:
        final = xarray.apply_ufunc(rio.fill.fillnodata, final, xarray.where(final == nodata, 0, 1), dask = 'parallelized', keep_attrs = 'override', kwargs = {'max_search_distance': fill_distance})
        final = persist(final)
    files = None
    # Clipping mosaic with vector mask
    if clip != None:
        if crs == None:
            crs = final.rio.crs
        shape = gpd.read_file(clip).to_crs(crs)
        shape = convert_3D_2D(shape)
        final = final.rio.clip(shape)
        final = persist(final)
    # Resampling to the same shape and resolution as another raster
    if reference_raster != None:
        with rioxarray.open_rasterio(reference_raster, chunks = True, lock = True) as tif:
            ref = tif.load()
        final = final.rio.reproject_match(ref)
        final = persist(final)
    # Because predictor = 2 works with float64 only when libtiff > 3.2.0 is installed and default libtiff in ubuntu is 3.2.0
    if final.dtype == 'float64':
        final = final.astype('float32')
    final.rio.to_raster(os.path.join(output_dir, band + '.tif'), compress = 'deflate', PREDICTOR = 2, ZLEVEL = 9, BIGTIFF = 'IF_SAFER', tiled = True, NUM_THREADS = 'NUM_CPUS', lock = True)
    return output_dir + band + '.tif'


def get_bands(paths, keep_all_channels):
    sets = []
    for path in paths:
        im_type = get_type(path)
        if im_type == 'Sentinel2_up':
            path = path + 'GRANULE/'
            path = glob(path + '*')[0]
            path = path + '/IMG_DATA/'
            if os.path.isdir(path + 'R10m/'):
                bands = glob(path + 'R10m/*B*.jp2') + glob(path + 'R20m/*B*.jp2') + glob(path + 'R60m/*B*.jp2')
            else:
                bands = glob(path + '*B*.jp2')
        else:
            bands = glob(path + '*.*[!(zip|tar|tar.gz|aux.xml)*]')
        sets.append([bands, im_type])
    # Getting imagery type and bands list
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
    sortedzeros = dict(sorted(zerosdict.items(), key=lambda item: item[1], reverse = False))
    order = list(sortedzeros.keys())
    return order
    
    
def ismultiband(inp):
    return os.path.isdir(inp)