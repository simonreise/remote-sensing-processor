import os
import pathlib
import numpy as np
from glob import glob
import xml.etree.ElementTree as ET

import geopandas as gpd
import shapely
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.merge
import rasterio.fill
from rasterio.io import MemoryFile
from rasterio.enums import Resampling

from remote_sensing_processor.imagery_types.types import get_type
from remote_sensing_processor.common.common_functions import get_resampling


def landsat_proc(path, projection, cloud_mask, pansharpen, keep_pan_band, resample, t, clipper):
    t = t.lower()
    bands = glob(path + '*B*.tif')
    resample = get_resampling(resample)
    # deleting gm bands for landsat 7
    bands = [band for band in bands if '_GM_' not in band]
    lsver = get_type(path)
    mtl = glob(path + '*MTL.xml')[0]
    mtl = ET.parse(mtl).getroot()
    # reading quality assessment band
    if cloud_mask == True:
        qa = glob(path + '*QA*')
        if len(qa) == 1:
            qa = qa[0]
            with rio.open(qa) as landsat_qa_file:
                qa = landsat_qa_file.read(1)
                qatransform = landsat_qa_file.transform
        else:
            qar = sorted(qa)[1]
            qa = sorted(qa)[0]
            with rio.open(qa) as landsat_qa_file:
                qa = landsat_qa_file.read(1)
                qatransform = landsat_qa_file.transform
            with rio.open(qar) as landsat_qar_file:
                qar = landsat_qar_file.read(1)
        if lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']:
            mask = np.where(qa == 21824, 0, 1)
        elif lsver in ['Landsat7_up_l1', 'Landsat7_up_l2', 'Landsat5_up_l1', 'Landsat5_up_l2']:
            mask = np.where(qa == 5440, 0, 1)
        elif lsver in ['Landsat1_up_l1', 'Landsat1_up_l2']:
            mask = np.where(qa == 256, 0, 1)
        if 'qar' in locals():
            mask = np.where(qar == 0, mask, 1)
        qa = None
        qar = None
    # picking temperature bands
    tbands = []
    for band in bands:
        if (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B10' in band) or ('B11' in band)):
            tbands.append(band)
        elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B6' in band) or ('B06' in band)):
            tbands.append(band)
        elif (lsver in ['Landsat5_up_l1', 'Landsat5_up_l2']) and (('B6' in band) or ('B06' in band)):
            tbands.append(band)
    # picking pansharpening band and pre-calculating coefficients
    if pansharpen == True and lsver in ['Landsat8_up_l1', 'Landsat7_up_l1']:
        for band in bands:
            if (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2', 'Landsat7_up_l1', 'Landsat7_up_l2']) and (('B8' in band) or ('B08' in band)):
                panband = band
        with rio.open(panband) as b:
            pan = b.read()
            pan_trans = b.transform
            pan_res = b.res
        for band in bands:
            if (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B4' in band) or ('B04' in band)):
                red = upscale(band, pan_trans, pan_res, pan.shape, resample)
            elif (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B3' in band) or ('B03' in band)):
                green = upscale(band, pan_trans, pan_res, pan.shape, resample)
            elif (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B2' in band) or ('B02' in band)):
                blue = upscale(band, pan_trans, pan_res, pan.shape, resample)
            elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B3' in band) or ('B03' in band)):
                red = upscale(band, pan_trans, pan_res, pan.shape, resample)
            elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B2' in band) or ('B02' in band)):
                green = upscale(band, pan_trans, pan_res, pan.shape, resample)
            elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B1.' in band) or ('B01' in band)):
                blue = upscale(band, pan_trans, pan_res, pan.shape, resample)
            elif (lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B4' in band) or ('B04' in band)):
                nir = upscale(band, pan_trans, pan_res, pan.shape, resample)
        if lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']:
            with np.errstate(invalid='ignore', divide='ignore'):
                pan = pan / ((0.42 * blue + 0.98 * green + 0.6 * red) / 2)
        elif lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']:
            with np.errstate(invalid='ignore', divide='ignore'):
                pan = pan / ((0.42 * blue + 0.98 * green + 0.6 * red + 1 * nir) / 3)
        red = None
        green = None
        blue = None
        nir = None
    # deleting pansharpening band if not needed
    if keep_pan_band == False:
        bands = [band for band in bands if ('B8' not in band) and ('B08' not in band)]
    #reading files
    outfiles = []
    for band in bands:
        with rio.open(band) as b:
            img = b.read()
            meta = b.profile
            bounds = b.bounds
            transform = b.transform
            resolution = b.res
            try:
                nodata = b.nodata
            except:
                nodata = 0
        #masking clouds
        if cloud_mask == True:
            if ('B8' in band) or ('B08' in band):
                mask1 = np.zeros((img.shape), mask.dtype)
                reproject(
                    source=mask,
                    destination=mask1,
                    src_transform=qatransform,
                    src_crs=meta['crs'],
                    dst_transform=transform,
                    dst_resolution = resolution,
                    dst_crs=meta['crs'],
                    resampling=Resampling.nearest)
                mask = mask1
                img = np.where(mask == 1, nodata, img)
            else:
                img = np.where(mask == 1, nodata, img)
        # DOS1 atmospheric correction
        if band not in tbands:
            if lsver in ['Landsat8_up_l1', 'Landsat7_up_l1', 'Landsat5_up_l1', 'Landsat1_up_l1']:
                btitle = band.split('B')[-1].split('.')[0]
                radM = float(mtl.findall('LEVEL1_MIN_MAX_RADIANCE/RADIANCE_MAXIMUM_BAND_' + btitle)[0].text)
                refM = float(mtl.findall('LEVEL1_MIN_MAX_REFLECTANCE/REFLECTANCE_MAXIMUM_BAND_' + btitle)[0].text)
                eSD = float(mtl.findall('IMAGE_ATTRIBUTES/EARTH_SUN_DISTANCE')[0].text)
                sE = float(mtl.findall('IMAGE_ATTRIBUTES/SUN_ELEVATION')[0].text)
                m = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_MULT_BAND_' + btitle)[0].text)
                a = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_ADD_BAND_' + btitle)[0].text)
                if lsver == 'Landsat8_up_l1':
                    eS = (np.pi * eSD * eSD) *radM / refM
                else:
                    # Esun from Chander, G.; Markham, B. L. & Helder, D. L. Summary of current radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1 ALI sensors Remote Sensing of Environment, 2009, 113, 893 - 903
                    # landsat 1
                    if 'LM01' in band:
                        dEsunB = {'ESUN_BAND1': 1823, 'ESUN_BAND2': 1559, 'ESUN_BAND3': 1276, 'ESUN_BAND4': 880.1}
                    # landsat 2
                    elif 'LM02' in band:
                        dEsunB = {'ESUN_BAND1': 1829, 'ESUN_BAND2': 1539, 'ESUN_BAND3': 1268, 'ESUN_BAND4': 886.6}	
                    # landsat 3
                    elif 'LM03' in band:
                        dEsunB = {'ESUN_BAND1': 1839, 'ESUN_BAND2': 1555, 'ESUN_BAND3': 1291, 'ESUN_BAND4': 887.9}
                    # landsat 4
                    elif 'LM04' in band:
                        dEsunB = {'ESUN_BAND1': 1827, 'ESUN_BAND2': 1569, 'ESUN_BAND3': 1260, 'ESUN_BAND4': 866.4}
                    elif 'LT04' in band:
                        dEsunB = {'ESUN_BAND1': 1983, 'ESUN_BAND2': 1795, 'ESUN_BAND3': 1539, 'ESUN_BAND4': 1028, 'ESUN_BAND5': 219.8, 'ESUN_BAND7': 83.49}
                    # landsat 5
                    elif 'LM05' in band:
                        dEsunB = {'ESUN_BAND1': 1824, 'ESUN_BAND2': 1570, 'ESUN_BAND3': 1249, 'ESUN_BAND4': 853.4}
                    elif 'LT05' in band:
                        dEsunB = {'ESUN_BAND1': 1983, 'ESUN_BAND2': 1796, 'ESUN_BAND3': 1536, 'ESUN_BAND4': 1031, 'ESUN_BAND5': 220, 'ESUN_BAND7': 83.44}
                    # landsat 7 Esun from http://landsathandbook.gsfc.nasa.gov/data_prod/prog_sect11_3.html
                    elif 'LE07' in band:
                        dEsunB = {'ESUN_BAND1': 1970, 'ESUN_BAND2': 1842, 'ESUN_BAND3': 1547, 'ESUN_BAND4': 1044, 'ESUN_BAND5': 225.7, 'ESUN_BAND7': 82.06, 'ESUN_BAND8': 1369}
                    eS = float(dEsunB['ESUN_BAND' + btitle])
                #sine sun elevation
                sA = np.sin(sE * np.pi /180)
                #dn 1%
                DNm = 0
                values, count  = np.unique(img, return_counts = True)
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
                # path radiance Lh = ML* DNm + AL  – 0.01* ESUNλ * cosθs / (π * d^2)
                Llmin = m * LDNm + a
                L1 = 0.01 * eS * sA / (np.pi * eSD * eSD)
                Lh = Llmin - L1
                # land surface reflectance ρ = [π * (Lλ - Lp) * d^2]/ (ESUNλ * cosθs)
                img = np.where(img == nodata, nodata, np.clip((((img * m + a) - Lh) * np.pi * eSD * eSD) / (eS * sA), 0, 1))
            elif lsver in ['Landsat8_up_l2', 'Landsat7_up_l2', 'Landsat5_up_l2', 'Landsat1_up_l2']:
                m = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/REFLECTANCE_MULT_BAND_' + btitle)[0].text)
                a = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/REFLECTANCE_ADD_BAND_' + btitle)[0].text)
                img = np.where(img == nodata, nodata, np.clip((img * m + a), 0, 1))
        #temperature
        if band in tbands:
            if t == 'c':
                deg = 273.15
            elif t == 'k':
                deg = 0
            btitle = band.split('B')[-1].split('.')[0]
            if lsver in ['Landsat5_up_l1', 'Landsat7_up_l1', 'Landsat8_up_l1']:
                mult = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_MULT_BAND_' + btitle)[0].text)
                add = float(mtl.findall('LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_ADD_BAND_' + btitle)[0].text)
                k1 = float(mtl.findall('LEVEL1_THERMAL_CONSTANTS/K1_CONSTANT_BAND_' + btitle)[0].text)
                k2 = float(mtl.findall('LEVEL1_THERMAL_CONSTANTS/K2_CONSTANT_BAND_' + btitle)[0].text)
                with np.errstate(invalid='ignore'):
                    img = np.where(img == nodata, nodata, (k2 / np.log(k1/((img * mult) + add) + 1)) - deg)
            elif lsver in ['Landsat5_up_l2', 'Landsat7_up_l2', 'Landsat8_up_l2']:
                mult = float(mtl.findall('LEVEL2_SURFACE_TEMPERATURE_PARAMETERS/TEMPERATURE_MULT_BAND_ST_B' + btitle)[0].text)
                add = float(mtl.findall('LEVEL2_SURFACE_TEMPERATURE_PARAMETERS/TEMPERATURE_ADD_BAND_ST_B' + btitle)[0].text)
                img = np.where(img == nodata, nodata, ((img * mult) + add) - deg)
        #pansharpening
        if pansharpen == True and lsver in ['Landsat8_up_l1' 'Landsat7_up_l1']:
            if ((lsver in ['Landsat8_up_l1', 'Landsat8_up_l2']) and (('B4' in band) or ('B04' in band) or ('B3' in band) or ('B03' in band) or  ('B2' in band) or ('B02' in band)))or ((lsver in ['Landsat7_up_l1', 'Landsat7_up_l2']) and (('B4' in band) or ('B04' in band) or ('B3' in band) or ('B03' in band) or ('B2' in band) or ('B02' in band) or ('B1.' in band) or ('B01' in band))):
                img1 = np.zeros(pan.shape, img.dtype) 
                reproject(
                    source=img,
                    destination=img1,
                    src_transform=meta['transform'],
                    src_crs=meta['crs'],
                    dst_transform=pan_trans,
                    dst_resolution = pan_res,
                    dst_crs=meta['crs'],
                    resampling=resample)
                img = img1 * pan
            elif (lsver in ['Landsat8_up_l1', 'Landsat8_up_l2', 'Landsat7_up_l1', 'Landsat7_up_l2']) and (('B8' not in band) and ('B08' not in band)):
                img1 = np.zeros(pan.shape) 
                reproject(
                    source=img,
                    destination=img1,
                    src_transform=meta['transform'],
                    src_crs=meta['crs'],
                    dst_transform=pan_trans,
                    dst_resolution = pan_res,
                    dst_crs=meta['crs'],
                    resampling=resample)
                img = img1
            trans = pan_trans
        else:
            trans = meta['transform']
            #reprojecting
        if projection != None:
            transform, width, height = calculate_default_transform(
                meta['crs'], projection, meta['width'], meta['height'], *bounds)
            proj = np.zeros((img.shape[0], height, width), img.dtype)
            img, transform = reproject(
                source=img,
                destination=proj,
                src_transform=trans,
                src_crs=meta['crs'],
                dst_transform=transform,
                dst_crs=projection,
                resampling=Resampling.nearest)
            img = proj
        else:
            projection = meta['crs']
        #clipping
        if clipper != None:
            shape = gpd.read_file(clipper).to_crs(crs)
            shape = convert_3D_2D(shape)
            with MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=img.shape[1],
                    width=img.shape[2],
                    count=img.shape[0],
                    dtype=img.dtype,
                    compress = 'lzw',
                    crs=projection,
                    transform=transform,
                    BIGTIFF='YES',
                    nodata = 0
                ) as temp:
                    temp.write(img)
                    img, transform = rio.mask.mask(temp, shape, crop=True, filled=True)
        #save
        bname = 'B' + band.split('B')[-1]
        pathres = path + bname
        outfiles.append(pathlib.Path(pathres))
        with rio.open(
            pathres,
            'w',
            driver='GTiff',
            height=img.shape[1],
            width=img.shape[2],
            count=img.shape[0],
            dtype=img.dtype,
            compress = 'deflate',
            PREDICTOR = 1,
            ZLEVEL=9,
            crs=projection,
            transform=transform,
            nodata = 0
        ) as outfile:
            outfile.write(img)
    for file in glob(path + '*'):
        file = pathlib.Path(file)
        if file not in outfiles:
            os.remove(file)


def upscale(band, pan_trans, pan_res, pan_shape, resample):
    with rio.open(band) as b:
        band = b.read()
        trans = b.transform
        crs = b.crs
    band1 = np.zeros(pan_shape) 
    reproject(
        source=band,
        destination=band1,
        src_transform=trans,
        src_crs=crs,
        dst_transform=pan_trans,
        dst_resolution = pan_res,
        dst_crs=crs,
        resampling=resample)
    return band1