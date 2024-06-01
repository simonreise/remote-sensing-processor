from glob import glob
import shutil
import os
import pathlib
import warnings
from typing import Any, List, Union, Optional
import rasterio as rio

from remote_sensing_processor.common.torch_test import cuda_test

from remote_sensing_processor.unzip.unzip import unzip_sentinel, unzip_landsat

from remote_sensing_processor.sentinel2.sen2cor.sen2cor import sen2correct
from remote_sensing_processor.sentinel2.superres.superres import superresolution
from remote_sensing_processor.sentinel2.sentinel_postprocessing.sentinel_postprocessing import (
    s2postprocess_superres,
    s2postprocess_no_superres,
)

from remote_sensing_processor.landsat.landsat import landsat_proc, del_landsat_temp

from remote_sensing_processor.mosaic.mosaic import mosaic_main, order, ismultiband

from remote_sensing_processor.indices.normalized_difference import nd
from remote_sensing_processor.imagery_types.types import get_type, get_index

from remote_sensing_processor.common.normalize import normalize_file

from remote_sensing_processor.common.replace import replace_val

from remote_sensing_processor.common.rasterize import rasterize_vector

from remote_sensing_processor.common.common_functions import get_first_proj
# A function that get dask client from a cluster, or creates local cluster.
# Will be useful if functions will support computation on dask clusters.
# TODO : make everything support computation on dask clusters
#from remote_sensing_processor.common.dask import get_client

from remote_sensing_processor import segmentation


__version__ = '0.2.2'

def sentinel2(
    archives: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
    sen2cor: Optional[bool] = True,
    upscale: Optional[str] = 'superres',
    resample: Optional[str] = 'bilinear',
    crs: Optional[Any] = None,
    cloud_mask: Optional[bool] = True,
    clip: Optional[Union[str, pathlib.Path]] = None,
    normalize: Optional[bool] = False,
) -> List[str]:
    """
    Preprocess Sentinel-2 imagery.
    
    Parameters
    ----------
    archives : string or list of strings
        Path to archive / directory or list of pathes to archives / directories.
    sen2cor : bool (default = True)
        Is atmospheric correction using Sen2Cor needed. Set to False if you have troubles with Sen2Cor.
    upscale : string or None (default = 'superres')
        Method for upscaling 20- and 60-m bands to 10 m resolution.
        Can be 'superres' - uses neural network for superresolution, 'resample' - uses resampling,
        or None - keeps original band resolution.
        Set it to 'resample' or None if you do not have GPU that supports CUDA.
    resample : resampling method from rasterio as a string (default = 'bilinear')
        Resampling method that will be used to upscale 20 and 60 m bands if upscale == 'resample'.
        You can read more about resampling methods
        `here <https://rasterio.readthedocs.io/en/latest/topics/resampling.html>`_.
    crs : string (optional)
        CRS in which output data should be or `same` to get CRS from the first archive.
    cloud_mask : bool (default = True)
        Is cloud masking needed.
    clip : string (optional)
        Path to vector file to be used to crop the image.
    normalize : bool (default = False)
        Is min-max normalization to 0-1 range needed.
        
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
    # Type checking
    if isinstance(archives, (str, pathlib.Path)):
        archives = [archives]
    elif isinstance(archives, list):
        for i in archives:
            if not isinstance(i, (str, pathlib.Path)):
                raise TypeError("archives must be a string or a list of strings")
    else:
        raise TypeError("archives must be a string or a list of strings")
    for archive in archives:
        if not os.path.exists(archive):
            raise OSError(archive + " does not exist")
    if not isinstance(sen2cor, bool):
        if sen2cor is None:
            sen2cor = True
        else:
            raise TypeError("sen2cor must be boolean")
    if upscale not in ['superres', 'resample'] and upscale is not None:
        raise ValueError("upscale must be 'superres', 'resample' or None")
    if not isinstance(resample, str):
        if resample is None:
            resample = 'bilinear'
        else:
            raise TypeError("resample must be a string")
    if crs is not None and not crs == 'same':
        rio.crs.CRS.from_user_input(crs)
    if not isinstance(cloud_mask, bool):
        if cloud_mask is None:
            cloud_mask = True
        else:
            raise TypeError("cloud_mask must be boolean")
    if clip is not None:
        if not isinstance(clip, (str, pathlib.Path)):
            raise TypeError("clip must be a string")
        elif not os.path.exists(clip):
            raise OSError(clip + " does not exist")
    if not isinstance(normalize, bool):
        if normalize is None:
            normalize = False
        else:
            raise TypeError("normalize must be boolean")
    
    cuda = cuda_test()
    if cuda == False and upscale == 'superres':
        warnings.warn('CUDA or MPS is not available. Superresolution process could be very slow.')
    
    paths = []
    for archive in archives:
        path = unzip_sentinel(archive)
        path1 = glob(path + '/*/')[0]
        if sen2cor == True:
            sen2correct(path1)
            path1 = glob(path + '/*/')[0]
        if upscale == 'superres':
            img = superresolution(input_dir=path1, clip=clip)
            if crs == 'same':
                crs = img.rio.crs
            s2postprocess_superres(
                img=img, 
                projection=crs, 
                cloud_mask=cloud_mask, 
                clip=clip, 
                normalize=normalize, 
                path=path, 
                path1=path1,
            )
        else:
            if crs == 'same':
                img = glob(path1 + '/**/*.jp2')[0]
                crs = get_first_proj(img)
            s2postprocess_no_superres(
                projection=crs, 
                cloud_mask=cloud_mask, 
                clip=clip, 
                normalize=normalize, 
                resample=resample, 
                path=path, 
                path1=path1, 
                upscale=upscale,
            )
        shutil.rmtree(path1)
        paths.append(path)
        print('Preprocessing of ' + str(archive) + ' completed')
    return paths
    

def landsat(
    archives: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
    crs: Optional[Any] = None,
    cloud_mask: Optional[bool] = True,
    pansharpen: Optional[bool] = True,
    keep_pan_band: Optional[bool] = False,
    resample: Optional[str] = 'bilinear',
    clip: Optional[Union[str, pathlib.Path]] = None,
    t: Optional[str] = 'k',
    normalize_t: Optional[bool] = False,
) -> List[str]:
    """
    Preprocess Landsat imagery.
    
    Parameters
    ----------
    archives : string or list of strings
        Path to archive / directory or list of pathes to archives / directories.
    crs : string (optional)
        CRS in which output data should be or `same` to get CRS from the first archive.
    cloud_mask : bool (default = True)
        Is cloud masking needed.
    pansharpen : bool (default = True)
        Is pansharpening needed. RSP uses Brovey transform for pansarpening Landsat 7, 8 and 9.
    keep_pan_band : bool (default = False)
        Keep pansharpening band or delete it. Pansharpening band have the same wavelengths as optical bands,
        so it does not contain any additional information to other bands. Affects only Landsat 7, 8 and 9.
    resample : resampling method from rasterio as a string (default = 'bilinear')
        Resampling method that will be used to upscale bands that cannot be upscaled in pansharpening operation.
        You can read more about resampling methods
        `here <https://rasterio.readthedocs.io/en/latest/topics/resampling.html>`_. Affects only Landsat 7, 8 and 9.
    clip : string (optional)
        Path to vector file to be used to crop the image.
    t : string ('k' or 'c', default = 'k')
        Convert thermal band to kelvins or celsius (no farenheit lol).
    normalize_t : bool (default = False)
        If True then thermal bands will be min-max normalized to 0-1 range.
    
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
    # Type checking
    if isinstance(archives, (str, pathlib.Path)):
        archives = [archives]
    elif isinstance(archives, list):
        for i in archives:
            if not isinstance(i, (str, pathlib.Path)):
                raise TypeError("archives must be a string or a list of strings")
    else:
        raise TypeError("archives must be a string or a list of strings")
    for archive in archives:
        if not os.path.exists(archive):
            raise OSError(archive + " does not exist")
    if crs is not None and not crs == 'same':
        rio.crs.CRS.from_user_input(crs)
    if not isinstance(cloud_mask, bool):
        if cloud_mask is None:
            cloud_mask = True
        else:
            raise TypeError("cloud_mask must be boolean")
    if not isinstance(pansharpen, bool):
        if pansharpen is None:
            pansharpen = True
        else:
            raise TypeError("pansharpen must be boolean")
    if not isinstance(keep_pan_band, bool):
        if keep_pan_band is None:
            keep_pan_band = False
        else:
            raise TypeError("keep_pan_band must be boolean")
    if not isinstance(resample, str):
        if resample is None:
            resample = 'bilinear'
        else:
            raise TypeError("resample must be a string")
    if t not in ['k', 'c']:
        if t is None:
            t = 'k'
        else:
            raise ValueError("t must be 'k' or 'c'")
    if clip is not None:
        if not isinstance(clip, (str, pathlib.Path)):
            raise TypeError("clip must be a string")
        elif not os.path.exists(clip):
            raise OSError(clip + " does not exist")
    if not isinstance(normalize_t, bool):
        if normalize_t is None:
            normalize_t = False
        else:
            raise TypeError("normalize_t must be boolean")
    
    paths = []
    for archive in archives:
        path = unzip_landsat(archive)
        if crs == 'same':
            crs = get_first_proj(glob(path + '/*.tif')[0])
        outfiles = landsat_proc(
            path=path, 
            projection=crs, 
            cloud_mask=cloud_mask, 
            pansharpen=pansharpen, 
            keep_pan_band=keep_pan_band, 
            resample=resample, 
            t=t, 
            clip=clip, 
            normalize_t=normalize_t,
        )
        del_landsat_temp(path, outfiles)
        paths.append(path)
        print('Preprocessing of ' + str(archive) + ' completed')
    return paths


def mosaic(
    inputs: List[Union[str, pathlib.Path]],
    output_dir: Union[str, pathlib.Path],
    fill_nodata: Optional[bool] = False,
    fill_distance: Optional[int] = 250,
    clip: Optional[Union[str, pathlib.Path]] = None,
    crs: Optional[Any] = None,
    nodata: Optional[Union[int, float]] = None,
    reference_raster: Optional[Union[str, pathlib.Path]] = None,
    resample: Optional[str] = 'average',
    nodata_order: Optional[bool] = False,
    match_hist: Optional[bool] = False,
    keep_all_channels: Optional[bool] = True,
) -> List[str]:
    """
    Creates mosaic from several rasters.
    
    Parameters
    ----------
    inputs : list of strings
        List of pathes to rasters to be merged or to folders where multiband imagery data is stored
        in order from images that should be on top to images that should be on bottom.
    output_dir: path to output directory as a string
        Path where mosaic raster or rasters will be saved.
    fill_nodata : bool (default = False)
        Is filling the gaps in the raster needed.
    fill_distance : int (default = 250)
        Fill distance for `fill_nodata` function.
    clip : string (optional)
        Path to vector file to be used to crop the image.
    crs : string (optional)
        CRS in which output data should be.
    nodata : int or float (default = None)
        Nodata value. If not set then is read from file or set to 0.
    reference_raster : path to reference raster as a string (optional)
        Reference raster is needed to bring output mosaic raster to same resolution and projection as other data source.
        Is useful when you need to use data from different sources together.
    resample : resampling method from rasterio as a string (default = 'average')
        Resampling method that will be used to reshape to a reference raster shape.
        You can read more about resampling methods
        `here <https://rasterio.readthedocs.io/en/latest/topics/resampling.html>`_.
        Use 'nearest' if you want to keep only class values.
    nodata_order : bool (default = False)
        Is needed to merge images in order from images with less nodata on top (they are usually clear)
        to images with most nodata values on bottom (they usually are most distorted and cloudy).
    match_hist : bool (default = False)
        Is needed to match histograms of merged images. Improve mosaic uniformity, but change original data.
    keep_all_channels : bool (default = True)
        Is needed only when you are merging Landsat images from different generations.
        If True, all bands are processed, if False, only bands that are present in all input images are processed
        and others are omited.
    
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
        >>> mosaic_sentinel = rsp.mosaic(input_sentinels, 
        ...     '/home/rsp_test/mosaics/sentinel/', 
        ...     clip=border, 
        ...     crs='EPSG:4326', 
        ...     nodata_order=True,
        ... )
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
        >>> mosaic_landcover = rsp.mosaic(
        ...     lcs, 
        ...     '/home/rsp_test/mosaics/landcover/', 
        ...     clip=border, 
        ...     reference_raster='/home/rsp_test/mosaics/sentinel/B1.tif', 
        ...     nodata=-1,
        ... )
        Processing completed
        >>> print(mosaic_landcover)
        ['/home/rsp_test/mosaics/landcover/ESA_WorldCover_10m_2020_v100_N60E075_Map_mosaic.tif']
    """
    # Type checking
    if isinstance(inputs, list):
        for i in inputs:
            if not isinstance(i, (str, pathlib.Path)):
                raise TypeError("inputs must be a list of strings")
    else:
        raise TypeError("inputs must be a list of strings")
    for inp in inputs:
        if not os.path.exists(inp):
            raise OSError(inp + " does not exist")
    if not isinstance(output_dir, (str, pathlib.Path)):
        raise TypeError("output_dir must be a string")
    if not isinstance(fill_nodata, bool):
        if fill_nodata is None:
            fill_nodata = False
        else:
            raise TypeError("fill_nodata must be boolean")
    if not isinstance(fill_distance, int):
        if fill_distance is None:
            fill_distance = 250
        else:
            raise TypeError("fill_distance must be an integer")
    if clip is not None:
        if not isinstance(clip, (str, pathlib.Path)):
            raise TypeError("clip must be a string")
        elif not os.path.exists(clip):
            raise OSError(clip + " does not exist")
    if crs is not None:
        rio.crs.CRS.from_user_input(crs)
    if not isinstance(nodata, (int, float)) and nodata is not None:
        raise TypeError("nodata must be integer or float")
    if reference_raster is not None:
        if not isinstance(reference_raster, (str, pathlib.Path)):
            raise TypeError("reference_raster must be a string")
        elif not os.path.exists(reference_raster):
            raise OSError(reference_raster + " does not exist")
    if not isinstance(resample, str):
        if resample is None:
            resample = 'average'
        else:
            raise TypeError("resample must be a string")
    if not isinstance(nodata_order, bool):
        if nodata_order is None:
            nodata_order = False
        else:
            raise TypeError("nodata_order must be boolean")
    if not isinstance(match_hist, bool):
        if match_hist is None:
            match_hist = False
        else:
            raise TypeError("match_hist must be boolean")
    if not isinstance(keep_all_channels, bool):
        if keep_all_channels is None:
            keep_all_channels = True
        else:
            raise TypeError("keep_all_channels must be boolean")
    
    mb = ismultiband(inputs[0])
    if mb == True:
        for i in range(len(inputs)):
            if not inputs[i].endswith(r'/'):
                inputs[i] = inputs[i] + r'/'
    if not output_dir.endswith(r'/'):
        output_dir = output_dir + r'/'
    if nodata_order == True:
        inputs = order(inputs)
    paths = mosaic_main(
        inputs=inputs, 
        output_dir=output_dir, 
        fill_nodata=fill_nodata, 
        fill_distance=fill_distance, 
        clip=clip, 
        crs=crs, 
        nodata=nodata, 
        reference_raster=reference_raster, 
        resample=resample, 
        match_hist=match_hist, 
        mb=mb, 
        keep_all_channels=keep_all_channels,
    )
    return paths


def calculate_index(
        name: str,
        folder: Optional[Union[str, pathlib.Path]] = None,
        b1: Optional[Union[str, pathlib.Path]] = None,
        b2: Optional[Union[str, pathlib.Path]] = None,
) -> str:
    """
    Calculates vegetation indexes.
    
    Parameters
    ----------
    name : string
        Name of index.
    folder: path to input product as a string (optional)
        If you define path to a supported imagery product and a name of supported index,
        you do not need to define `b1` and `b2`. Bands needed for index calculation are picked automatically.
    b1, b2 : path as string (optional)
        Path to band to calculate normalized difference index. If you define bands, you do not need to define `folder`,
        but still need to define `name` - it will be an output file name.
    
    Returns
    ----------
    string
        Path where index raster is saved.
    
    Examples
    --------
        >>> ndvi = rsp.calculate_index(
        ...     'NDVI', 
        ...     '/home/rsp_test/mosaics/sentinel/'
        ... )
        >>> print(ndvi)
        '/home/rsp_test/mosaics/sentinel/NDVI.tif'
        
        >>> ndvi = rsp.calculate_index(
        ...     'NDVI', 
        ...     b1='/home/rsp_test/mosaics/sentinel/B8.tif', 
        ...     b2='/home/rsp_test/mosaics/sentinel/B4.tif',
        ... )
        >>> print(ndvi)
        '/home/rsp_test/mosaics/sentinel/NDVI.tif'
    """
    # Type checking
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if folder is not None:
        if not isinstance(folder, (str, pathlib.Path)):
            raise TypeError("folder must be a string")
        elif not os.path.exists(folder):
            raise OSError(folder + " does not exist")
    if b1 is not None:
        if not isinstance(b1, (str, pathlib.Path)):
            raise TypeError("b1 must be a string")
        elif not os.path.exists(b1):
            raise OSError(b1 + " does not exist")
    if b2 is not None:
        if not isinstance(b2, (str, pathlib.Path)):
            raise TypeError("b2 must be a string")
        elif not os.path.exists(b2):
            raise OSError(b2 + " does not exist")
    
    if (folder is not None) and ((b1 is None) or (b2 is None)):
        if not folder.endswith(r'/'):
            folder = folder + r'/'
        t = get_type(folder)
        if t == 'Undefined':
            raise ValueError('Cannot define imagery type')
        b1, b2 = get_index(t, name, folder)
    if (b1 is not None) and (b2 is not None):
        path = nd(name=name, b1=b1, b2=b2, folder=folder)
    else:
        raise ValueError('Bands 1 and 2 must be defined')
    return path


def normalize(
    input_file: Union[str, pathlib.Path],
    output_file: Union[str, pathlib.Path],
    minimum: Optional[Union[int, float]] = None,
    maximum: Optional[Union[int, float]] = None,
) -> None:
    """
    Applies min-max normalization to input file.
    
    Parameters
    ----------
    input_file : string
        Path to input file.
    output_file : string
        Path to output file.
    minimum: int or float (optional)
        Min value for normalization. If not defined then min and max of data type of `input_file` will be used.
    maximum: int or float (optional)
        Max value for normalization. If not defined then min and max of data type of `input_file` will be used.
    
    Examples
    --------
        >>> rsp.normalize(
        ...     '/home/rsp_test/mosaics/sentinel/B1.tif', 
        ...     '/home/rsp_test/mosaics/sentinel/B1_norm.tif', 
        ...     0, 
        ...     10000,
        ... )
    """
    # Type checking
    if not isinstance(input_file, (str, pathlib.Path)):
        raise TypeError("input_file must be a string")
    elif not os.path.exists(input_file):
        raise OSError(input_file + " does not exist")
    if not isinstance(output_file, (str, pathlib.Path)):
        raise TypeError("output_file must be a string")
    if not isinstance(minimum, (int, float)) and minimum is not None:
        raise TypeError("minimum must be int or float")
    if not isinstance(maximum, (int, float)) and maximum is not None:
        raise TypeError("maximum must be int or float")
    assert minimum < maximum
    
    normalize_file(input_file, output_file, minimum, maximum)
    
    
def replace_value(
    input_file: Union[str, pathlib.Path],
    output_file: Union[str, pathlib.Path],
    old: Union[int, float],
    new: Union[int, float],
) -> None:
    """
    Replaces a specific value in a raster.
    
    Parameters
    ----------
    input_file : string
        Path to input file.
    output_file : string
        Path to output file.
    old: int or float
        An old value to replace.
    new: int or float
        A new value to insert.
    
    Examples
    --------
        >>> rsp.replace_value(
        ...     '/home/rsp_test/mosaics/sentinel/B1.tif', 
        ...     '/home/rsp_test/mosaics/sentinel/B1_new.tif', 
        ...     0, 
        ...     -9999,
        ... )
    """
    # Type checking
    if not isinstance(input_file, (str, pathlib.Path)):
        raise TypeError("input_file must be a string")
    elif not os.path.exists(input_file):
        raise OSError(input_file + " does not exist")
    if not isinstance(output_file, (str, pathlib.Path)):
        raise TypeError("output_file must be a string")
    if not isinstance(old, (int, float)):
        raise TypeError("old must be int or float")
    if not isinstance(new, (int, float)):
        raise TypeError("new must be int or float")
    
    replace_val(input_file, output_file, new, old, nodata=False)
    
    
def replace_nodata(
    input_file: Union[str, pathlib.Path],
    output_file: Union[str, pathlib.Path],
    new: Union[int, float],
    old: Optional[Union[int, float]] = None,
) -> None:
    """
    Replaces a nodata value in a raster.
    
    Parameters
    ----------
    input_file : string
        Path to input file.
    output_file : string
        Path to output file.
    new: int or float
        A new nodata value to insert.
    old: int or float (optional)
        An old nodata value to replace. If not stated then is read from input file.
    
    Examples
    --------
        >>> rsp.replace_nodata(
        ...     '/home/rsp_test/mosaics/landcover/landcover.tif', 
        ...     '/home/rsp_test/mosaics/landcover/landcover_new.tif', 
        ...     0,
        ... )
    """
    #type checking
    if not isinstance(input_file, (str, pathlib.Path)):
        raise TypeError("input_file must be a string")
    elif not os.path.exists(input_file):
        raise OSError(input_file + " does not exist")
    if not isinstance(output_file, (str, pathlib.Path)):
        raise TypeError("output_file must be a string")
    if not isinstance(old, (int, float)) and old is not None:
        raise TypeError("old must be int or float")
    if not isinstance(new, (int, float)):
        raise TypeError("new must be int or float")
    
    replace_val(input_file, output_file, new, old, nodata=True)
    
    
def rasterize(
    vector: Union[str, pathlib.Path],
    reference_raster: Union[str, pathlib.Path],
    value: str,
    output_file: Union[str, pathlib.Path],
    nodata: Optional[Union[int, float]] = 0,
) -> None:
    """
    Rasterizes a vector file.
    
    Parameters
    ----------
    vector : string
        Path to vector file that needs to be rasterized.
    reference_raster : path to reference raster as a string
        Path to a raster file to get shape, resolution and projection from.
    value : string
        A field to use for a burn-in value. Field should be numeric.
    output_file : string
        Path to output file.
    nodata: int or float (default = 0)
        A value that will be used as nodata.
    
    Examples
    --------
        >>> rsp.rasterize(
        ...     '/home/rsp_test/mosaics/treecover/treecover.shp', 
        ...     '/home/rsp_test/mosaics/sentinel/B1.tif', 
        ...     'tree_species', 
        ...     '/home/rsp_test/mosaics/treecover/treecover.tif', 
        ...     nodata=0,
        ... )
    """
    # Type checking
    if not isinstance(vector, (str, pathlib.Path)):
        raise TypeError("vector must be a string")
    elif not os.path.exists(vector):
        raise OSError(vector + " does not exist")
    if not isinstance(reference_raster, (str, pathlib.Path)):
        raise TypeError("reference_raster must be a string")
    elif not os.path.exists(reference_raster):
        raise OSError(reference_raster + " does not exist")
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if not isinstance(output_file, (str, pathlib.Path)):
        raise TypeError("output_file must be a string")
    if not isinstance(nodata, (int, float)):
        if nodata is None:
            nodata = 0
        else:
            raise TypeError("new must be int or float")
    
    rasterize_vector(vector, reference_raster, value, output_file, nodata)