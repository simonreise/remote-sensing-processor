import os
import pathlib
from glob import glob
import xml.etree.ElementTree as ET
from functools import partial
import warnings

import numpy as np
import xarray
import dask
#from dask.distributed import Lock

import geopandas as gpd
from rasterio.enums import Resampling
import rioxarray

from remote_sensing_processor.imagery_types.types import get_type
from remote_sensing_processor.common.common_functions import get_resampling, convert_3D_2D, persist


warnings.filterwarnings("ignore", message="divide by zero")
warnings.filterwarnings("ignore", message="invalid value encountered")

def landsat_proc(path, 
    projection, 
    cloud_mask, 
    pansharpen, 
    keep_pan_band, 
    resample, 
    t, 
    clip, 
    normalize_t,
):
    t = t.lower()
    bands = glob(path + '*B*.tif')
    if bands == []:
        bands = glob(path + '*B*.TIF')
    # Removing quality band from bands list if it is there
    bands = [x for x in bands if not 'BQA' in x]
    resample = get_resampling(resample)
    # Deleting gm bands for landsat 7
    bands = [band for band in bands if '_GM_' not in band]
    band_names = [('B' + band.split('B')[-1].split('.')[0]) for band in bands]
    lsver = get_type(path)
    # Getting metadata
    mtl = glob(path + '*MTL.xml')
    if mtl != []:
        mtl = mtl[0]
        mtl = ET.parse(mtl).getroot()
    else:
        mtl = glob(path + '*MTL.txt')[0]
        with open(mtl) as file:
            mtl = [line.rstrip() for line in file]
    # Reading quality assessment band
    if cloud_mask == True:
        qa = glob(path + '*QA*')
        if len(qa) == 1:
            qa = qa[0]
            # Old qa images have different values
            if 'BQA' in qa:
                old = True
            else:
                old = False
            with rioxarray.open_rasterio(qa, chunks=True, lock=True) as tif:
                qa = persist(tif)#.load().chunk('auto')
            qar = None
        else:
            qar = sorted(qa)[1]
            qa = sorted(qa)[0]
            # Old qa images have different values
            if 'BQA' in qa:
                old = True
            else:
                old = False
            with rioxarray.open_rasterio(qa, chunks=True, lock=True) as tif:
                qa = persist(tif)#.load().chunk('auto')
            with rioxarray.open_rasterio(qar, chunks=True, lock=True) as tif:
                qar = persist(tif)#.load().chunk('auto')
        clears = []
        for v in np.unique(qa):
            val = format(v, 'b')[::-1]
            while not len(val) == 16:
                val = val + '0'
            # Collection 1 qa images have different values
            if old:
                if lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']:
                    if (
                        val[0] == '0' 
                        and val[1] == '0' 
                        and val[2:4] == '00' 
                        and val[4] == '0' 
                        and val[5:7] in ['00', '10'] 
                        and val[7:9] in ['00', '10'] 
                        and val[11:13] in ['00', '10']
                    ):
                        clears.append(v)
                elif lsver in ['Landsat7_up_l1', 'Landsat7_up_l2', 'Landsat5_up_l1', 'Landsat5_up_l2']:
                    if (
                        val[0] == '0' 
                        and val[1] == '0' 
                        and val[2:4] == '00' 
                        and val[4] == '0' 
                        and val[5:7] in ['00', '10'] 
                        and val[7:9] in ['00', '10']
                    ):
                        clears.append(v)
                elif lsver in ['Landsat1_up_l1', 'Landsat1_up_l2']:
                    if (
                        val[0] == '0' 
                        and val[1] == '0' 
                        and val[2:4] == '00' 
                        and val[4] == '0' 
                        and val[5:7] in ['00', '10']
                    ):
                        clears.append(v)
            else:
                if lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']:
                    if (
                        val[0] == '0' 
                        and val[1] == '0' 
                        and val[2] == '0' 
                        and val[3] == '0' 
                        and val[4] == '0' 
                        and val[6] == '1' 
                        and val[8:10] in ['00', '10'] 
                        and val[10:12] in ['00', '10'] 
                        and val[14:16] in ['00', '10']
                    ):
                        clears.append(v)
                elif lsver in ['Landsat7_up_l1', 'Landsat7_up_l2', 'Landsat5_up_l1', 'Landsat5_up_l2']:
                    if (
                        val[0] == '0' 
                        and val[1] == '0' 
                        and val[3] == '0' 
                        and val[4] == '0' 
                        and val[8:10] in ['00', '10'] 
                        and val[10:12] in ['00', '10']
                    ):
                        clears.append(v)
                elif lsver in ['Landsat1_up_l1', 'Landsat1_up_l2']:
                    if (
                        val[0] == '0' 
                        and val[3] == '0' 
                        and val[8:10] in ['00', '10']
                    ):
                        clears.append(v)
        mask = xarray.where(qa.isin(clears), 0, 1)
        if not isinstance(qar, type(None)):
            mask = mask.where(qar == 0, 1)
    else:
        mask = None
    # Picking temperature bands
    tbands = []
    for band in band_names:
        if (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B10' in band) or ('B11' in band)):
            tbands.append(band)
        elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B6' in band) or ('B06' in band)):
            tbands.append(band)
        elif (lsver in ['Landsat5_up_l1', 'Landsat5_up_l2']) and (('B6' in band) or ('B06' in band)):
            tbands.append(band)
    # Picking pansharpening band and bands that can be pansharpened
    if pansharpen == True and lsver in ['Landsat8_up_l1', 'Landsat8_up_l2', 'Landsat7_up_l1', 'Landsat7_up_l2']:
        for i in range(len(bands)):
            if ((lsver in ['Landsat8_up_l1', 'Landsat8_up_l2', 'Landsat7_up_l1', 'Landsat7_up_l2'])
                    and (('B8' in band_names[i]) or ('B08' in band_names[i]))
            ):
                break
        panband = bands.pop(i)
        panband_name = band_names.pop(i)
        for band in band_names:
            if (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B4' in band) or ('B04' in band)):
                red = band
            elif (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B3' in band) or ('B03' in band)):
                green = band
            elif (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B2' in band) or ('B02' in band)):
                blue = band
            elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B3' in band) or ('B03' in band)):
                red = band
            elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B2' in band) or ('B02' in band)):
                green = band
            elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B1.' in band) or ('B01' in band)):
                blue = band
            elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B4' in band) or ('B04' in band)):
                nir = band
    else:
        pan = None
        red = None
        green = None
        blue = None
        nir = None
    # Reading files
    files = []
    for i in bands:
        with rioxarray.open_rasterio(i, chunks=True, lock=True) as tif:
            files.append(persist(tif))#.load().chunk('auto'))
        #tif.close()
    img = xarray.concat(files, dim=xarray.Variable('band', band_names))
    #img = xarray.concat([rioxarray.open_rasterio(i, chunks=True, lock=Lock("rio-read", client=client)) for i in bands], dim=xarray.Variable('band', band_names))
    try:
        if img.rio.nodata == None:
            nodata = 0
        else:
            nodata = img.rio.nodata
    except:
        nodata = 0
    img.rio.write_nodata(nodata, inplace = True)
    img = persist(img)
    # Masking clouds
    if cloud_mask == True:
        img = img.where((mask.squeeze()) == 0, nodata)
    # Converting dn to reflectance and temperature to degrees
    calc_p = partial(calc, tbands=tbands, lsver=lsver, mtl=mtl, t=t, normalize_t=normalize_t, nodata=nodata, path=path)
    img = img.groupby('band', squeeze=False).map(calc_p)
    img = persist(img)
    # Pansharpening
    if (((pansharpen == True) or (keep_pan_band == True))
            and (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2', 'Landsat7_up_l1', 'Landsat7_up_l2'])
            and (('B8' not in band) and ('B08' not in band))
    ):
        with rioxarray.open_rasterio(
                panband,
                chunks=True,
                lock=True
        ).drop_vars('band').assign_coords({'band': [panband_name]}) as tif:
            pan = persist(tif)#.load().drop_vars('band').assign_coords({'band': [panband_name]}).chunk('auto')
        pan = pan.where(((mask.rio.reproject_match(pan, resampling=Resampling.nearest)).squeeze()) == 0, nodata)
        pan = calc(pan, tbands, lsver, mtl, t, normalize_t, nodata, path)
        pan = persist(pan)
        if pansharpen == True:
            img = img.rio.reproject_match(pan, resampling=resample)
            img = img.chunk("auto")
            if lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']:
                with np.errstate(invalid='ignore', divide='ignore'):
                    pan_c = (pan / ((0.42 * img.sel({'band': blue})
                             + 0.98 * img.sel({'band': green})
                             + 0.6 * img.sel({'band': red})) / 2)).squeeze()
                    bands_to_pan = [b for b in img['band'].values if b in ['B4', 'B04', 'B3', 'B03', 'B2', 'B02']]
                #img = img.where(img == nodata | ~img['band'].isin(['B4', 'B04', 'B3', 'B03', 'B2', 'B02']), (img * pan_c.squeeze()).clip(0, 1))
            elif lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']:
                with np.errstate(invalid = 'ignore', divide = 'ignore'):
                    pan_c = (pan / ((0.42 * img.sel({'band': blue})
                                     + 0.98 * img.sel({'band': green})
                                     + 0.6 * img.sel({'band': red}) + 1 * img.sel({'band': nir})) / 3)).squeeze()
                    bands_to_pan = [b for b in img['band'].values if b in [
                        'B4', 'B04', 'B3', 'B03', 'B2', 'B02', 'B1', 'B01'
                    ]]
            pansharp_p = partial(pansharp, pan_c=pan_c, bands_to_pan=bands_to_pan, nodata=nodata)
            img = img.groupby('band', squeeze=False).map(pansharp_p)
            pan.rio.write_nodata(nodata, inplace=True)
            if keep_pan_band == True:
                img = xarray.concat([img, pan], dim='band')   
            img = persist(img)
    # Reprojecting
    if projection != None:
        img = img.rio.reproject(projection, resampling=Resampling.nearest)
        img = img.chunk("auto")
        if ((pansharpen == False) and (keep_pan_band == True)):
            pan = pan.rio.reproject(projection, resampling=Resampling.nearest)
            pan = pan.chunk("auto")
            pan = persist(pan)
        img = persist(img)
    else:
        projection = img.rio.crs
    # Clipping
    if clip != None:
        shape = gpd.read_file(clip).to_crs(projection)
        shape = convert_3D_2D(shape)
        img = img.rio.clip(shape.geometry.values, shape.crs)
        if ((pansharpen == False) and (keep_pan_band == True)):
            pan = pan.rio.clip(shape.geometry.values, shape.crs)
            pan = persist(pan)
        img = persist(img)
    # Because predictor = 2 works with float64 only when libtiff > 3.2.0 is installed and default libtiff in ubuntu is 3.2.0
    if img.dtype == 'float64':
        img = img.astype('float32')
        img = persist(img)
    # Save
    outfiles = []
    results = []
    for band in img:
        pathres = path + band['band'].item() + '.tif'
        outfiles.append(pathlib.Path(pathres))
        results.append(band.rio.to_raster(
            pathres, 
            compress='deflate', 
            PREDICTOR=2, 
            ZLEVEL=9, 
            BIGTIFF='IF_SAFER', 
            tiled=True, 
            NUM_THREADS='ALL_CPUS',
            compute=False, 
            lock=True,
        ))
    if ((pansharpen == False) and (keep_pan_band == True)):
        pathres = path + pan['band'][0].item() + '.tif'
        outfiles.append(pathlib.Path(pathres))
        results.append(pan.rio.to_raster(
            pathres, 
            compress='deflate', 
            PREDICTOR=2, 
            ZLEVEL=9, 
            BIGTIFF='IF_SAFER', 
            tiled=True, 
            NUM_THREADS='ALL_CPUS',
            compute=False, 
            lock=True,
        ))
    dask.compute(*results)
    # Unused code
    #print(dask.is_dask_collection(img))
    #save_p = partial(save, path = path)
    #outfiles = img.groupby('band').map(save_p)
    #results = []
    #for i in img:
        #results.append(dask.delayed(write)(i, path))
    #futures = dask.persist(*results)
    #outfiles = dask.compute(*futures)
    #img = client.scatter(img)
    #write_p = partial(write, path = path)
    #outfiles = img.groupby('band').map(write_p)
    #futs = client.map(write_p, img)
    #outfiles = client.gather(futs)
    # Process bands in parallel with multiprocessing
    #outfiles = []
    #for band in bands:
        #outfiles.append(band_proc(band = band, cloud_mask = cloud_mask, mask = mask, tbands = tbands, t = t, lsver = lsver, mtl = mtl, pansharpen = pansharpen, pan = pan, projection = projection, clip = clip, path = path, resample = resample, client = client))
    #bf = partial(band_proc, cloud_mask = cloud_mask, mask = mask, tbands = tbands, t = t, lsver = lsver, mtl = mtl, pansharpen = pansharpen, pan = pan, projection = projection, clip = clip, path = path, resample = resample)
    #outfutures = client.map(bf, bands, pure = False)
    #outfiles = []
    #for o in outfutures:
        #outfiles.append(o.result())
    #with multiprocessing.Pool() as pool:
        #bf = partial(band_proc, cloud_mask = cloud_mask, mask = mask, tbands = tbands, t = t, lsver = lsver, mtl = mtl, pansharpen = pansharpen, pan = pan, projection = projection, clip = clip, path = path, resample = resample)
        #outfiles = pool.map(bf, bands)
    return outfiles


def calc(img, tbands, lsver, mtl, t, normalize_t, nodata, path):
    # DOS1 atmospheric correction
    if img['band'] not in tbands:
        if lsver in ['Landsat8_up_l1', 'Landsat7_up_l1', 'Landsat5_up_l1', 'Landsat1_up_l1']:
            #btitle = band.split('B')[-1].split('.')[0]
            btitle = img['band'].item()[1:]
            if isinstance(mtl, ET.Element):
                radM = float(mtl.findall('LEVEL1_MIN_MAX_RADIANCE/RADIANCE_MAXIMUM_BAND_' + btitle)[0].text)
                refM = float(mtl.findall('LEVEL1_MIN_MAX_REFLECTANCE/REFLECTANCE_MAXIMUM_BAND_' + btitle)[0].text)
                eSD = float(mtl.findall('IMAGE_ATTRIBUTES/EARTH_SUN_DISTANCE')[0].text)
                sE = float(mtl.findall('IMAGE_ATTRIBUTES/SUN_ELEVATION')[0].text)
                m = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_MULT_BAND_' + btitle)[0].text)
                a = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_ADD_BAND_' + btitle)[0].text)
            else:
                radM = float([i for i in mtl if 'RADIANCE_MAXIMUM_BAND_' + btitle in i][0].split('=')[1])
                refM = float([i for i in mtl if 'REFLECTANCE_MAXIMUM_BAND_' + btitle in i][0].split('=')[1])
                eSD = float([i for i in mtl if 'EARTH_SUN_DISTANCE' in i][0].split('=')[1])
                sE = float([i for i in mtl if 'SUN_ELEVATION' in i][0].split('=')[1])
                m = float([i for i in mtl if 'RADIANCE_MULT_BAND_' + btitle in i][0].split('=')[1])
                a = float([i for i in mtl if 'RADIANCE_ADD_BAND_' + btitle in i][0].split('=')[1])
            if lsver == 'Landsat8_up_l1':
                eS = (np.pi * eSD * eSD) * radM / refM
            else:
                band = os.path.normpath(path).split(os.sep)[-1]
                # Esun from Chander, G.; Markham, B. L. & Helder, D. L.
                # Summary of current radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1 ALI sensors
                # Remote Sensing of Environment, 2009, 113, 893 - 903
                # Landsat 1
                if 'LM01' in band:
                    dEsunB = {'ESUN_BAND1': 1823, 'ESUN_BAND2': 1559, 'ESUN_BAND3': 1276, 'ESUN_BAND4': 880.1}
                # Landsat 2
                elif 'LM02' in band:
                    dEsunB = {'ESUN_BAND1': 1829, 'ESUN_BAND2': 1539, 'ESUN_BAND3': 1268, 'ESUN_BAND4': 886.6}	
                # Landsat 3
                elif 'LM03' in band:
                    dEsunB = {'ESUN_BAND1': 1839, 'ESUN_BAND2': 1555, 'ESUN_BAND3': 1291, 'ESUN_BAND4': 887.9}
                # Landsat 4
                elif 'LM04' in band:
                    dEsunB = {'ESUN_BAND1': 1827, 'ESUN_BAND2': 1569, 'ESUN_BAND3': 1260, 'ESUN_BAND4': 866.4}
                elif 'LT04' in band:
                    dEsunB = {'ESUN_BAND1': 1983, 'ESUN_BAND2': 1795, 'ESUN_BAND3': 1539, 'ESUN_BAND4': 1028,
                              'ESUN_BAND5': 219.8, 'ESUN_BAND7': 83.49}
                # Landsat 5
                elif 'LM05' in band:
                    dEsunB = {'ESUN_BAND1': 1824, 'ESUN_BAND2': 1570, 'ESUN_BAND3': 1249, 'ESUN_BAND4': 853.4}
                elif 'LT05' in band:
                    dEsunB = {'ESUN_BAND1': 1983, 'ESUN_BAND2': 1796, 'ESUN_BAND3': 1536, 'ESUN_BAND4': 1031,
                              'ESUN_BAND5': 220, 'ESUN_BAND7': 83.44}
                # Landsat 7 Esun from http://landsathandbook.gsfc.nasa.gov/data_prod/prog_sect11_3.html
                elif 'LE07' in band:
                    dEsunB = {'ESUN_BAND1': 1970, 'ESUN_BAND2': 1842, 'ESUN_BAND3': 1547, 'ESUN_BAND4': 1044,
                              'ESUN_BAND5': 225.7, 'ESUN_BAND7': 82.06, 'ESUN_BAND8': 1369}
                eS = float(dEsunB['ESUN_BAND' + btitle])
            # Sine sun elevation
            sA = np.sin(sE * np.pi /180)
            # dn 1%
            DNm = 0
            values, count = np.unique(img, return_counts=True)
            rasterBandUniqueVal = dict(zip(values, count))
            rasterBandUniqueVal.pop(0, None)
            sumTot = sum(rasterBandUniqueVal.values())
            pT1pc = sumTot * 0.0001
            newSum = 0
            for i in sorted(rasterBandUniqueVal):
                DNm = i
                newSum = newSum + rasterBandUniqueVal[i]
                if newSum >= pT1pc:
                    DNm = i
                    break
            LDNm = DNm
            # Path radiance Lh = ML* DNm + AL  – 0.01* ESUNλ * cosθs / (π * d^2)
            Llmin = m * LDNm + a
            L1 = 0.01 * eS * sA / (np.pi * eSD * eSD)
            Lh = Llmin - L1
            # Land surface reflectance ρ = [π * (Lλ - Lp) * d^2]/ (ESUNλ * cosθs)
            return img.where(img == nodata, ((((img * m + a) - Lh) * np.pi * eSD * eSD) / (eS * sA)).clip(0, 1))
        elif lsver in ['Landsat8_up_l2', 'Landsat7_up_l2', 'Landsat5_up_l2', 'Landsat1_up_l2']:
            if isinstance(mtl, ET.Element):
                m = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/REFLECTANCE_MULT_BAND_' + btitle)[0].text)
                a = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/REFLECTANCE_ADD_BAND_' + btitle)[0].text)
            else:
                m = float([i for i in mtl if 'REFLECTANCE_MULT_BAND_' + btitle in i][0].split('=')[1])
                a = float([i for i in mtl if 'REFLECTANCE_ADD_BAND_' + btitle in i][0].split('=')[1])
            return img.where(img == nodata, (img * m + a).clip(0, 1))
    # Temperature
    if img['band'] in tbands:
        if t == 'c':
            deg = 273.15
        elif t == 'k':
            deg = 0
        #btitle = band.split('B')[-1].split('.')[0]
        btitle = img['band'].item()[1:]
        if lsver in ['Landsat5_up_l1', 'Landsat7_up_l1', 'Landsat8_up_l1']:
            if isinstance(mtl, ET.Element):
                mult = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_MULT_BAND_' + btitle)[0].text)
                add = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_ADD_BAND_' + btitle)[0].text)
                k1 = float(mtl.findall('LEVEL1_THERMAL_CONSTANTS/K1_CONSTANT_BAND_' + btitle)[0].text)
                k2 = float(mtl.findall('LEVEL1_THERMAL_CONSTANTS/K2_CONSTANT_BAND_' + btitle)[0].text)
            else:
                mult = float([i for i in mtl if 'RADIANCE_MULT_BAND_' + btitle in i][0].split('=')[1])
                add = float([i for i in mtl if 'RADIANCE_ADD_BAND_' + btitle in i][0].split('=')[1])
                k1 = float([i for i in mtl if 'K1_CONSTANT_BAND_' + btitle in i][0].split('=')[1])
                k2 = float([i for i in mtl if 'K2_CONSTANT_BAND_' + btitle in i][0].split('=')[1])
            with np.errstate(invalid='ignore'):
                img = img.where(img == nodata, (k2 / np.log(k1/((img * mult) + add) + 1)) - deg)
        elif lsver in ['Landsat5_up_l2', 'Landsat7_up_l2', 'Landsat8_up_l2']:
            if isinstance(mtl, ET.Element):
                mult = float(mtl.findall('LEVEL2_SURFACE_TEMPERATURE_PARAMETERS/TEMPERATURE_MULT_BAND_ST_B' + btitle)[0].text)
                add = float(mtl.findall('LEVEL2_SURFACE_TEMPERATURE_PARAMETERS/TEMPERATURE_ADD_BAND_ST_B' + btitle)[0].text)
            else:
                mult = float([i for i in mtl if 'TEMPERATURE_MULT_BAND_ST_B' + btitle in i][0].split('=')[1])
                add = float([i for i in mtl if 'TEMPERATURE_ADD_BAND_ST_B' + btitle in i][0].split('=')[1])
            img = img.where(img == nodata, ((img * mult) + add) - deg)
        # Normalize temperature in range 175 k - 375 k 
        if normalize_t:
            img = img.where(img == nodata, (img - (175 - deg)) / ((375 - deg) - (175 - deg)))
        return img
            


def pansharp(img, pan_c, bands_to_pan, nodata):
    if img['band'] in bands_to_pan:
        return img.where(img == nodata, (img * pan_c).clip(0, 1))
    else:
        return img


def del_landsat_temp(path, outfiles):
    for file in glob(path + '*'):
        file = pathlib.Path(file)
        if file not in outfiles:
            os.remove(file)


# No longer needed
"""
def write(band, path):
    pathres = path + band['band'].item() + '.tif'
    band.rio.to_raster(pathres, compress='deflate', PREDICTOR=2, ZLEVEL=9, BIGTIFF='IF_SAFER', tiled=True, windowed=True) #, lock=threading.Lock())
    return pathlib.Path(pathres)
    
    
def upscale(band, pan, resample):
    with rioxarray.open_rasterio(band, chunks=True, lock=False) as tif:
        band = tif #.load()
    crs = band.rio.crs
    band = band.rio.reproject_match(pan, resampling=resample)
    return band"""