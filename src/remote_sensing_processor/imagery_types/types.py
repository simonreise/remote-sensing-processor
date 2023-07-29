import numpy as np
from glob import glob
import os
import re

def get_type(path):
    if os.path.isdir(path) == True:
        bands = glob(path + '*.*[!(zip|tar|tar.gz|aux.xml)*]')
        for i in range(len(bands)):
            bands[i] = os.path.basename(bands[i]).split('.')[0]
    else:
        bands = []
    if sorted([*set([s for s in bands if re.compile('B*\d').match(s)])]) == ['B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']:
        t = 'Sentinel2_p' #sentinel2 preprocessed in RSP
    elif re.search('T\d\d\w\w\w_', path) != None:
        t = 'Sentinel2_up' #sentinel2 without processing
    elif re.search('MTD_MSIL1C.xml', path) != None:
        t = 'Sentinel2_up' #sentinel2 without processing
    elif re.search('L\w\d\d', path):
        if re.search('L\w\d\d', path).group(0) in ['LM05', 'LM04', 'LM03', 'LM02', 'LM01']:
            if len(sorted(bands)[0]) <= 3:
                t = 'Landsat1_p' #landsat1 processed in RSP
            else:
                if re.search('L\d\w\w', path):
                    if re.search('L\d\w\w', path).group(0) in ['L1TP', 'L1GT', 'L1GS']:
                        t = 'Landsat1_up_l1' #landsat1 without processing level 1
                    elif re.search('L\d\w\w', path).group(0) in ['L2SP', 'L2SR']:
                        t = 'Landsat1_up_l2' #landsat1 without processing level 2
                else:
                    t = 'Undefined_Landsat1'
        elif re.search('L\w\d\d', path).group(0) in ['LT05', 'LT04']:
            if len(sorted(bands)[0]) <= 3:
                t = 'Landsat5_p' #landsat5 processed in RSP
            else:
                if re.search('L\d\w\w', path):
                    if re.search('L\d\w\w', path).group(0) in ['L1TP', 'L1GT', 'L1GS']:
                        t = 'Landsat5_up_l1' #landsat5 without processing level 1
                    elif re.search('L\d\w\w', path).group(0) in ['L2SP', 'L2SR']:
                        t = 'Landsat5_up_l2' #landsat5 without processing level 2
                else:
                    t = 'Undefined_Landsat5'
        elif re.search('L\w\d\d', path).group(0) == 'LE07':
            if len(sorted(bands)[0]) <= 3:
                t = 'Landsat7_p' #landsat7 processed in RSP
            else:
                if re.search('L\d\w\w', path):
                    if re.search('L\d\w\w', path).group(0) in ['L1TP', 'L1GT', 'L1GS']:
                        t = 'Landsat7_up_l1' #landsat7 without processing level 1
                    elif re.search('L\d\w\w', path).group(0) in ['L2SP', 'L2SR']:
                        t = 'Landsat7_up_l2' #landsat7 without processing level 2
                else:
                    t = 'Undefined_Landsat7'
        elif re.search('L\w\d\d', path).group(0) in ['LC08', 'LC09']:
            if len(sorted(bands)[0]) <= 3:
                t = 'Landsat8_p' #landsat8 processed in RSP
            else:
                if re.search('L\d\w\w', path):
                    if re.search('L\d\w\w', path).group(0) in ['L1TP', 'L1GT', 'L1GS']:
                        t = 'Landsat8_up_l1' #landsat8 without processing level 1
                    elif re.search('L\d\w\w', path).group(0) in ['L2SP', 'L2SR']:
                        t = 'Landsat8_up_l2' #landsat8 without processing level 2
                else:
                    t = 'Undefined_Landsat8'
    else:
        t = 'Undefined'
    return t
    
def get_index(t, index, folder):
    index = index.upper()
    if t == 'Sentinel2_up':
        folder = folder + 'GRANULE/'
        folder = glob(folder + '*')[0]
        folder = folder + '/IMG_DATA/'
        if os.path.isdir(folder + 'R10m/'):
            bands = glob(folder + 'R10m/*B*.jp2') + glob(folder + 'R20m/*B*.jp2') + glob(path + 'R60m/*B*.jp2')
        else:
            bands = glob(folder + '*B*.jp2')
    else:
        bands = glob(folder + '*.*')
    if index == 'NDVI':
        if t in ['Sentinel2_up', 'Sentinel2_p']:
            for band in bands:
                if ('B08.jp2' in band) or ('B08_10m' in band) or ('B8.tif' in band):
                    b1 = band
                elif ('B04.jp2' in band) or ('B04_10m' in band) or ('B4.tif' in band):
                    b2 = band
        elif t in ['Landsat8_up_l1', 'Landsat8_up_l2', 'Landsat8_p']:
            for band in bands:
                if ('B5' in band) or ('B05' in band):
                    b1 = band
                elif ('B4' in band) or ('B04' in band):
                    b2 = band
        elif t in ['Landsat7_up_l1', 'Landsat7_up_l2', 'Landsat7_p', 'Landsat5_up_l1', 'Landsat5_up_l2', 'Landsat5_p']:
            for band in bands:
                if ('B4' in band) or ('B04' in band):
                    b1 = band
                elif ('B3' in band) or ('B03' in band):
                    b2 = band
        elif t in ['Landsat1_up_l1', 'Landsat1_up_l2', 'Landsat1_p']:
            for band in bands:
                if ('B4' in band) or ('B04' in band):
                    b1 = band
                elif ('B2' in band) or ('B02' in band):
                    b2 = band
    bands = (b1, b2)
    return bands