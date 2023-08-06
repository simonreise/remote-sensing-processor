from glob import glob
import shutil
import os
import sys

from remote_sensing_processor.common.torch_test import cuda_test

from remote_sensing_processor.unzip.unzip import unzip_sentinel, unzip_landsat

from remote_sensing_processor.sentinel2.sen2cor.sen2cor import sen2correct
from remote_sensing_processor.sentinel2.superres.superres import Superresolution
from remote_sensing_processor.sentinel2.sentinel_postprocessing.sentinel_postprocessing import s2postprocess_superres, s2postprocess_no_superres, get_first_proj

from remote_sensing_processor.landsat.landsat import landsat_proc

from remote_sensing_processor.mosaic.mosaic import mosaic_main, order, ismultiband

from remote_sensing_processor.indices.normalized_difference import nd
from remote_sensing_processor.imagery_types.types import get_type, get_index

from remote_sensing_processor.common.normalize import normalize_file

from remote_sensing_processor import segmentation


__version__ = '0.2'

cuda_test()

def sentinel2(archives, sen2cor = True, superres = True, projection = None, cloud_mask = True, clipper = None):
    """
    Preprocess Sentinel-2 imagery
    
    Parameters
    ----------
    archives : string or list of strings
        Path to archive or list of pathes to archives.
    sen2cor : bool (default = True)
        Is atmospheric correction using Sen2Cor needed. Set to False if you have troubles with Sen2Cor.
    superres : bool (default = True)
        Is upscaling 20- and 60-m bands to 10 m resolution needed. Set to False if you do not have tensorflow-supported GPU.
    projection : string (optional)
        CRS in which output data should be.
    cloud_mask : bool (default = True)
        Is cloud masking needed.
    clipper : string (optional)
        Path to vector file to be used to crop the image.
        
    Returns
    ----------
    list of strings
        List of paths where preprocessed Sentinel-2 products are saved.
    
    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> from glob import glob
        >>> sentinel2_imgs = glob('/home/rsp_test/sentinels/*.zip')
        >>> print(sentinel2_imgs)
        ['/home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626.zip',
         '/home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626.zip',
         '/home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626.zip',
         '/home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027.zip',
         '/home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624.zip',
         '/home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041.zip']
        >>> output_sentinels = rsp.sentinel2(sentinel2_imgs)
        Preprocessing of /home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041.zip completed
        >>> print(output_sentinels)
        ['/home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626/',
         '/home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626/',
         '/home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626/',
         '/home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027/',
         '/home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624/',
         '/home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041/']
    """
    if isinstance(archives, str):
        archives = [archives]
    paths = []
    for archive in archives:
        path = unzip_sentinel(archive)
        path1 = glob(path+'*')[0]
        if sen2cor == True:
            sen2correct(path1)
        path1 = glob(path+'*')[0]
        if superres == True:
            Superresolution(input_dir = path1, output_dir = path1, copy_original_bands = True, clip_to_aoi = False, geometry = None, bounds = None).start()
            img = glob(path+'**/*_superresolution.tif')[0]
            if projection == 'same':
                projection = get_first_proj(img)
            s2postprocess_superres(img = img, projection = projection, cloud_mask = cloud_mask, clipper = clipper, path = path, path1 = path1)
        else:
            if projection == 'same':
                img = glob(path1 + '/**/*.jp2')[0]
                projection = get_first_proj(img)
            s2postprocess_no_superres(projection = projection, cloud_mask = cloud_mask, clipper = clipper, path = path, path1 = path1)
        shutil.rmtree(path1)
        paths.append(path)
        print('Preprocessing of ' + archive + ' completed')
    return paths
    

def landsat(archives, projection = None, cloud_mask = True, pansharpen = True, keep_pan_band = False, resample = 'bilinear', t = 'k', clipper = None):
    """
    Preprocess Landsat imagery
    
    Parameters
    ----------
    archives : string or list of strings
        Path to archive or list of pathes to archives.
    projection : string (optional)
        CRS in which output data should be.
    cloud_mask : bool (default = True)
        Is cloud masking needed.
    pansharpen : bool (default = True)
        Is pansharpening needed. RSP uses Brovey transform for pansarpening Landsat 7, 8 and 9.
    keep_pan_band : bool (default = False)
        Keep pansharpening band or delete it. Pansharpening band have the same wavelengths as optical bands, so it does not contain any additional information to other bands. Affects only Landsat 7, 8 and 9. 
    resample : resampling method from rasterio as a string (default = 'bilinear')
        Resampling method that will be used to upscale bands that cannot be upscaled in pansharpening operation. You can read more about resampling methods `here <https://rasterio.readthedocs.io/en/latest/topics/resampling.html>`_. Affects only Landsat 7, 8 and 9. 
    t : string ('k' or 'c', default = 'k')
        Convert thermal band to kelvins or celsius (no farenheit lol).
    clipper : string (optional)
        Path to vector file to be used to crop the image.
    
    Returns
    ----------
    list of strings
        List of paths where preprocessed Landsat products are saved.
        
    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> from glob import glob
        >>> landsat_imgs = glob('/home/rsp_test/landsat/*.tar')
        >>> print(landsat_imgs)
        ['/home/rsp_test/landsat/LC08_L1TP_160023_20210825_20210901_02_T1.tar',
         '/home/rsp_test/landsat/LT05_L1TP_160023_20110814_20200820_02_T1.tar',
         '/home/rsp_test/landsat/LE07_L1TP_159023_20210826_20210921_02_T1.tar',
         '/home/rsp_test/landsat/LT05_L1TP_162023_20110812_20200820_02_T1.tar',
         '/home/rsp_test/landsat/LM05_L1TP_161023_19930803_20211018_02_T2.tar']
        >>> output_landsats = rsp.landsat(landsat_imgs)
        Preprocessing of /home/rsp_test/landsat/LC08_L1TP_160023_20210825_20210901_02_T1.tar completed
        Preprocessing of /home/rsp_test/landsat/LT05_L1TP_160023_20110814_20200820_02_T1.tar completed
        Preprocessing of /home/rsp_test/landsat/LE07_L1TP_159023_20210826_20210921_02_T1.tar completed
        Preprocessing of /home/rsp_test/landsat/LT05_L1TP_162023_20110812_20200820_02_T1.tar completed
        Preprocessing of /home/rsp_test/landsat/LM05_L1TP_161023_19930803_20211018_02_T2.tar completed
        >>> print(output_landsats)
        ['/home/rsp_test/landsat/LC08_L1TP_160023_20210825_20210901_02_T1/',
         '/home/rsp_test/landsat/LT05_L1TP_160023_20110814_20200820_02_T1/',
         '/home/rsp_test/landsat/LE07_L1TP_159023_20210826_20210921_02_T1/',
         '/home/rsp_test/landsat/LT05_L1TP_162023_20110812_20200820_02_T1/',
         '/home/rsp_test/landsat/LM05_L1TP_161023_19930803_20211018_02_T2/']
    """
    if isinstance(archives, str):
        archives = [archives]
    paths = []
    for archive in archives:
        path = unzip_landsat(archive)
        landsat_proc(path = path, projection = projection, cloud_mask = cloud_mask, pansharpen = pansharpen, keep_pan_band = keep_pan_band, resample = resample, t = t, clipper = clipper)
        paths.append(path)
        print('Preprocessing of ' + archive + ' completed')
    return paths


def mosaic(inputs, output_dir, fill_nodata = False, fill_distance = 250, clipper = None, crs = None, nodata = None, reference_raster = None, resample = 'average', nodata_order = False, keep_all_channels = True):
    """
    Creates mosaic from several rasters.
    
    Parameters
    ----------
    inputs : list of strings
        List of pathes to rasters to be merged or to folders where multiband imagery data is stored.
    output_dir: path to output directory as a string
        Path where mosaic raster or rasters will be saved.
    fill_nodata : bool (default = False)
        Is filling the gaps in the raster needed.
    fill_distance : int (default = 250)
        Fill distance for `fill_nodata` function.
    clipper : string (optional)
        Path to vector file to be used to crop the image.
    crs : string (optional)
        CRS in which output data should be.
    nodata : int or float (default = None)
        Nodata value. If not set then is read from file or set to 0.
    reference_raster : path to reference raster as a string (optional)
        Reference raster is needed to bring output mosaic raster to same resolution and projection as other data source. Is useful when you need to use data from different sources together.
    resample : resampling method from rasterio as a string (default = 'average
        Resampling method that will be used to reshape to a reference raster shape. You can read more about resampling methods `here <https://rasterio.readthedocs.io/en/latest/topics/resampling.html>`_. Use 'nearest' if you want to keep only class values.
    nodata_order : bool (default = False)
        Is needed to merge images in order from images with most nodata values on bottom (they usually are most distorted and cloudy) to images with less nodata on top (they are usually clear).
    keep_all_channels : bool (default = True)
        Is needed only when you are merging Landsat images from different generations. If True, all bands are processed, if False, only bands that are present in all input images are processed and others are omited.
    
    Returns
    ----------
    list of strings
        List of paths to mosaic rasters.
        
    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> input_sentinels = ['/home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626/',
        ...                    '/home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626/',
        ...                    '/home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626/',
        ...                    '/home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027/',
        ...                    '/home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624/',
        ...                    '/home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041/']
        >>> border = '/home/rsp_test/border.gpkg'
        >>> mosaic_sentinel = rsp.mosaic(input_sentinels, '/home/rsp_test/mosaics/sentinel/', clipper = border, projection = 'EPSG:4326', nodata_order = True)
        Processing completed
        >>> print(mosaic_sentinel)
        ['/home/rsp_test/mosaics/sentinel/B1.tif',
         '/home/rsp_test/mosaics/sentinel/B2.tif',
         '/home/rsp_test/mosaics/sentinel/B3.tif',
         '/home/rsp_test/mosaics/sentinel/B4.tif',
         '/home/rsp_test/mosaics/sentinel/B5.tif',
         '/home/rsp_test/mosaics/sentinel/B6.tif',
         '/home/rsp_test/mosaics/sentinel/B7.tif',
         '/home/rsp_test/mosaics/sentinel/B8.tif',
         '/home/rsp_test/mosaics/sentinel/B8A.tif',
         '/home/rsp_test/mosaics/sentinel/B9.tif',
         '/home/rsp_test/mosaics/sentinel/B11.tif',
         '/home/rsp_test/mosaics/sentinel/B12.tif']
        
        >>> lcs = glob('/home/rsp_test/landcover/*.tif')
        >>> print(lcs)
        ['/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N60E075_Map.tif',
         '/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N63E072_Map.tif',
         '/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N63E075_Map.tif']
        >>> mosaic_landcover = rsp.mosaic(lcs, '/home/rsp_test/mosaics/landcover/', clipper = border, reference_raster = '/home/rsp_test/mosaics/sentinel/B1.tif', nodata = -1)
        Processing completed
        >>> print(mosaic_landcover)
        ['/home/rsp_test/mosaics/landcover/ESA_WorldCover_10m_2020_v100_N60E075_Map_mosaic.tif']
    """
    mb = ismultiband(inputs[0])
    if mb == True:
        for i in range(len(inputs)):
            if not inputs[i].endswith(r'/'):
                inputs[i] = inputs[i] + r'/'
    if not output_dir.endswith(r'/'):
        output_dir = output_dir + r'/'
    if nodata_order == True:
        inputs = order(inputs)
    paths = mosaic_main(inputs = inputs, output_dir = output_dir, fill_nodata = fill_nodata, fill_distance = fill_distance, clipper = clipper, crs = crs, nodata = nodata, reference_raster = reference_raster, resample = resample, mb = mb, keep_all_channels = keep_all_channels)
    return paths

def calculate_index(name, folder = None, b1 = None, b2 = None):
    """
    Calculates vegetation indexes.
    
    Parameters
    ----------
    name : string
        Name of index.
    folder: path to input product as a string (optional)
        If you define path to a supported imagery product and a name of supported index, you do not need to define `b1` and `b2`. Bands needed for index calculation are picked automatically.
    b1, b2 : path as string (optional)
        Path to band to calculate normalized difference index. If you define bands, you do not need to define `folder`, but still need to define `name` - it will be an output file name.
    
    Returns
    ----------
    string
        Path where index raster is saved.
    
    Examples
    --------
        >>> ndvi = rsp.calculate_index('NDVI', '/home/rsp_test/mosaics/sentinel/')
        >>> print(ndvi)
        '/home/rsp_test/mosaics/sentinel/NDVI.tif'
        
        >>> ndvi = rsp.calculate_index('NDVI', b1 = '/home/rsp_test/mosaics/sentinel/B8.tif', b2 = '/home/rsp_test/mosaics/sentinel/B4.tif')
        >>> print(ndvi)
        '/home/rsp_test/mosaics/sentinel/NDVI.tif'
    """
    if (folder != None) and ((b1 == None) or (b2 == None)):
        if not folder.endswith(r'/'):
            folder = folder + r'/'
        t = get_type(folder)
        if t == 'Undefined':
            raise ValueError('Cannot define imagery type')
        b1, b2 = get_index(t, name, folder)
    if (b1 != None) and (b2 != None):
        path = nd(name = name, b1 = b1, b2 = b2, folder = folder)
    else:
        raise ValueError('Bands 1 and 2 must be defined')
    return path


def normalize(input_file, output_file, minimum = None, maximum = None):
    """
    Applies min-max normalization to input file.
    
    Parameters
    ----------
    input_file : string
        Path to input file.
    output_file : string
        Path to output file.
    min: int or float (optional)
        Min value for normalization. If not defined then min and max of data type of `input_file` will be used.
    max: int or float (optional)
        Max value for normalization. If not defined then min and max of data type of `input_file` will be used.
    
    Examples
    --------
        >>> rsp.normalize('/home/rsp_test/mosaics/sentinel/B1.tif', '/home/rsp_test/mosaics/sentinel/B1_norm.tif', 0, 10000)
    """
    normalize_file(input_file, output_file, minimum, maximum)