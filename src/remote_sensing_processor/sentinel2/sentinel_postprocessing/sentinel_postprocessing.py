import numpy as np
from glob import glob

import geopandas as gpd
import shapely
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.merge
import rasterio.fill
from rasterio.io import MemoryFile
from rasterio.enums import Resampling

from remote_sensing_processor.common.common_functions import convert_3D_2D


def s2postprocess_superres(img, projection, cloud_mask, clipper, path, path1):
    with rio.open(img) as bands:
        img = bands.read()
        meta = bands.profile
        bounds = bands.bounds
        descs = bands.descriptions
        transform = bands.transform
    #masking zeros
    img = np.where(img == 0, 1, img)
    #masking 65555 pixels
    mask = np.where(img > 15000, 0, 1)
    for i in range(img.shape[0]):
        img[i] = rio.fill.fillnodata(img[i], mask[i], max_search_distance = 500)
    if cloud_mask == True:
        #masking clouds by mask
        try:
            shape = glob(path1 + '/GRANULE/**/QI_DATA/MSK_CLOUDS_B00.gml')[0]
            cloudmask = gpd.read_file(clipper).to_crs(crs)
            cloudmask = convert_3D_2D(cloudmask)
            #with fiona.open(shape, "r") as shapefile:
            #    cloudmask = [feature["geometry"] for feature in shapefile]
            with MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=img.shape[1],
                    width=img.shape[2],
                    count=img.shape[0],
                    dtype=img.dtype,
                    compress = 'lzw',
                    crs=meta['crs'],
                    transform=transform
                ) as temp:
                    temp.write(img)
                    img, transform = rio.mask.mask(temp, cloudmask, invert = True, filled = True)
        except:
            pass
        #masking clouds by algorythm
        l1path = glob(path1 + r'/GRANULE/*/IMG_DATA/R20m/*SCL_20m.jp2')
        with rio.open(l1path[0]) as classification:
            cls = classification.read()
            clsmeta = classification.profile
        mask = np.empty_like(img[0])
        reproject(
            cls, mask,
            src_transform = clsmeta['transform'],
            dst_transform = meta['transform'],
            src_crs = clsmeta['crs'],
            dst_crs = meta['crs'],
            resampling = Resampling.nearest)
        for i in range(len(descs)):
            if 'B1 ' in descs[i]:
                band1 = i
        for i in range(len(descs)):
            if 'B9 ' in descs[i]:
                band9 = i
        clmask = np.where(((img[band1] >= 3200) & (img[band9] >= 6000) & (mask >= 7)), 1, 0)
        #delete 1px gaps in mask
        clmask = np.where((rio.fill.fillnodata(np.where(clmask == 1, 0, 1), np.where(clmask == 1, 0, 1), max_search_distance = 2)) == 1, 0, 1)
        #emerge mask
        clmask = rio.fill.fillnodata(clmask, clmask, max_search_distance = 22)
        img = np.where(clmask > 0, 0, img)
        #masking nodata
        img = np.where(mask == 0, 0, img)
    if projection != None:
        #reprojecting
        transform, width, height = calculate_default_transform(
            meta['crs'], projection, meta['width'], meta['height'], *bounds)
        proj = np.zeros((img.shape[0], height, width), img.dtype)
        img, transform = reproject(
            source=img,
            destination=proj,
            src_transform=meta['transform'],
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
    for i in range(img.shape[0]):
        pathres = path + '/' + descs[i].split(' ')[1] + '.tif'
        with rio.open(
            pathres,
            'w',
            driver='GTiff',
            height=img.shape[1],
            width=img.shape[2],
            count=1,
            dtype=img.dtype,
            compress = 'deflate',
            PREDICTOR = 2,
            ZLEVEL=9,
            crs=projection,
            transform=transform,
            nodata = 0
        ) as outfile:
            outfile.write(img[i],1)
            
            
            
def s2postprocess_no_superres(projection, cloud_mask, clipper, path, path1):
    bandnames = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    bands = []
    for b in bandnames:
        band = glob(path1 + '/**/*' + b + '_10m.jp2', recursive = True) 
        if band != []:
            bands.append(band[0])
        else:
            band = glob(path1 + '/**/*' + b + '_20m.jp2', recursive = True)
            if band != []:
                bands.append(band[0])
            else: 
                band = glob(path1 + '/**/*' + b + '_60m.jp2', recursive = True)
                if band != []:
                    bands.append(band[0])
    bandoutnames = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    for i in range(len(bands)):
        with rio.open(bands[i]) as band:
            img = band.read()
            meta = band.profile
            bounds = band.bounds
            descs = band.descriptions
            transform = band.transform
        #masking zeros
        img = np.where(img == 0, 1, img)
        #masking 65555 pixels
        mask = np.where(img > 15000, 0, 1)
        for j in range(img.shape[0]):
            img[j] = rio.fill.fillnodata(img[j], mask[j], max_search_distance = 500)
        if cloud_mask == True:
            #masking clouds by mask
            try:
                shape = glob(path1 + '/GRANULE/**/QI_DATA/MSK_CLOUDS_B00.gml')[0]
                cloudmask = gpd.read_file(clipper).to_crs(crs)
                cloudmask = convert_3D_2D(cloudmask)
                #with fiona.open(shape, "r") as shapefile:
                #    cloudmask = [feature["geometry"] for feature in shapefile]
                with MemoryFile() as memfile:
                    with memfile.open(
                        driver='GTiff',
                        height=img.shape[1],
                        width=img.shape[2],
                        count=img.shape[0],
                        dtype=img.dtype,
                        compress = 'lzw',
                        crs=meta['crs'],
                        transform=transform
                    ) as temp:
                        temp.write(img)
                        img, transform = rio.mask.mask(temp, cloudmask, invert = True, filled = True)
            except:
                pass
            #masking clouds by algorythm
            l1path = glob(path1 + r'/GRANULE/*/IMG_DATA/R20m/*SCL_20m.jp2')
            with rio.open(l1path[0]) as classification:
                cls = classification.read()
                clsmeta = classification.profile
            mask = np.empty_like(img[0])
            reproject(
                cls, mask,
                src_transform = clsmeta['transform'],
                dst_transform = meta['transform'],
                src_crs = clsmeta['crs'],
                dst_crs = meta['crs'],
                resampling = Resampling.nearest) 
            with rio.open(bands[0]) as b1:
                band1 = b1.read()
                b1meta = b1.profile
            b1 = np.empty_like(img[0])
            reproject(
                band1, b1,
                src_transform = b1meta['transform'],
                dst_transform = meta['transform'],
                src_crs = b1meta['crs'],
                dst_crs = meta['crs'],
                resampling = Resampling.nearest)
            with rio.open(bands[9]) as b9:
                band9 = b9.read()
                b9meta = b9.profile
            b9 = np.empty_like(img[0])
            reproject(
                band9, b9,
                src_transform = b9meta['transform'],
                dst_transform = meta['transform'],
                src_crs = b9meta['crs'],
                dst_crs = meta['crs'],
                resampling = Resampling.nearest)
            clmask = np.where(((b1 >= 3200) & (b9 >= 6000) & (mask >= 7)), 1, 0)
            #delete 1px gaps in mask
            clmask = np.where((rio.fill.fillnodata(np.where(clmask == 1, 0, 1), np.where(clmask == 1, 0, 1), max_search_distance = 2)) == 1, 0, 1)
            #emerge mask
            clmask = rio.fill.fillnodata(clmask, clmask, max_search_distance = 22)
            img = np.where(clmask > 0, 0, img)
            #masking nodata
            img = np.where(mask == 0, 0, img)
        if projection != None:
            #reprojecting
            transform, width, height = calculate_default_transform(
                meta['crs'], projection, meta['width'], meta['height'], *bounds)
            proj = np.zeros((img.shape[0], height, width), img.dtype)
            img, transform = reproject(
                source=img,
                destination=proj,
                src_transform=meta['transform'],
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
        pathres = path + '/' + bandoutnames[i] + '.tif'
        with rio.open(
            pathres,
            'w',
            driver='GTiff',
            height=img.shape[1],
            width=img.shape[2],
            count=1,
            dtype=img.dtype,
            compress = 'deflate',
            PREDICTOR = 2,
            ZLEVEL=9,
            crs=projection,
            transform=transform,
            nodata = 0
        ) as outfile:
            outfile.write(img[0],1)



def get_first_proj(img):
    with rio.open(img) as im:
        projection = im.crs
    return projection