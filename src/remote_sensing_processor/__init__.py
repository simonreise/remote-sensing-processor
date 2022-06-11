from glob import glob
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from remote_sensing_processor.unzip.unzip import unzip_sentinel, unzip_landsat

from remote_sensing_processor.sentinel2.sen2cor.sen2cor import sen2correct
from remote_sensing_processor.sentinel2.superres.superres import Superresolution
from remote_sensing_processor.sentinel2.sentinel_postprocessing.sentinel_postprocessing import s2postprocess_superres, s2postprocess_no_superres, get_first_proj

from remote_sensing_processor.landsat.landsat import landsat_proc

from remote_sensing_processor.mosaic.mosaic import mosaic_main, order, ismultiband

from remote_sensing_processor.indices.normalized_difference import nd
from remote_sensing_processor.imagery_types.types import get_type, get_index

from remote_sensing_processor.postprocessing.tiles import get_tiles, predict_map


__version__ = '0.1'

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


def mosaic(inputs, output_dir, fill_nodata = False, fill_distance = 250, clipper = None, crs = None, nodata = 0, reference_raster = None, nodata_order = False, keep_all_channels = True):
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
    nodata : int or float (default = 0)
        Nodata value.
    reference_raster : path to reference raster as a string (optional)
        Reference raster is needed to bring output mosaic raster to same resolution and projection as other data source. Is useful when you need to use data from different sources together.
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
    paths = mosaic_main(inputs = inputs, output_dir = output_dir, fill_nodata = fill_nodata, fill_distance = fill_distance, clipper = clipper, crs = crs, nodata = nodata, reference_raster = reference_raster, mb = mb, keep_all_channels = keep_all_channels)
    return paths

def normalized_difference(name, folder = None, b1 = None, b2 = None):
    """
    Calculates normalized difference indexes.
    
    Parameters
    ----------
    name : string
        Name of index.
    folder: path to input product as a string (optional)
        If you define path to a supported imagery product and a name of supported index, you do not need to define `b1` and `b2`. Bands needed for index calculation are picked automatically.
    b1, b2 : paths as strings
        Paths to bands to calculate normalized difference index. If you define bands, you do not need to define `folder`, but still need to define `name` - it will be an output file name.
    
    Returns
    ----------
    string
        Path where index raster is saved.
    
    Examples
    --------
        >>> ndvi = rsp.normalized_difference('NDVI', '/home/rsp_test/mosaics/sentinel/')
        >>> print(ndvi)
        '/home/rsp_test/mosaics/sentinel/NDVI.tif'
        
        >>> ndvi = rsp.normalized_difference('NDVI', b1 = '/home/rsp_test/mosaics/sentinel/B8.tif', b2 = '/home/rsp_test/mosaics/sentinel/B4.tif')
        >>> print(ndvi)
        '/home/rsp_test/mosaics/sentinel/NDVI.tif'
    """
    if (folder != None) and ((b1 == None) or (b2 == None)):
        if not folder.endswith(r'/'):
            folder = folder + r'/'
        t = get_type(folder)
        try:
            if t == 'Undefined':
                raise ValueError('Cannot define imagery type')
        except ValueError as e:
            print(e)
            sys.exit(1)
        b1, b2 = get_index(t, name, folder)
    try:
        if (b1 != None) and (b2 != None):
            path = nd(name = name, b1 = b1, b2 = b2, folder = folder)
        else:
            raise ValueError('Bands 1 and 2 must be defined')
    except ValueError as e:
        print(e)
        sys.exit(1)
    return path
        
             
def generate_tiles(x, y, tile_size = 128, categorical = True, num_classes = None, shuffle = False, samples_file = None, split = [1], x_outputs = None, y_outputs = None, dtype = None, nodata = None):
    """
    Cut rasters into tiles
    
    Parameters
    ----------
    x : list of paths as strings
        Rasters to use as training data.
    y : path as a string
        Raster to use as target values.
    tile_size : int
        Size of tiles to generate (tile_size x tile_size).
    categorical : bool (default = True)
        If y data is categorical. Usually True for classification and segmentation tasks and False for regression tasks.
    num_classes: int (optional)
        Number of classes in categorical y data.
    shuffle : bool (default = False)
        Is random shuffling of samples needed.
    samples_file : path as a string (optional)
        Path where to save tiles and samples data that are generated as output. File should have .pickle format. It can be later needed for mapping.
    split : list of ints (optional)
        Splitting data in subsets. Is a list of integers defining proportions of every subset. [3, 1, 1] will generate 3 subsets in proportion 3 to 1 to 1.
    x_outputs : list of paths as strings (optional)
        List of paths to save generated output x data. Data is saved in numpy .npy format.
    y_outputs : list of paths as strings (optional)
        List of paths to save generated output y data. Data is saved in numpy .npy format.
    dtype : dtype definition as a string (optional)
        If you run out of memory, you can try to convert your data to less memory consuming format.
    nodata : int or float (optional)
        If you want to ignore tiles that contain only nodata, you can define which value in y raster corresponds to nodata and tiles that contain only nodata will be omited.
    
    Returns
    ----------
    tuple:
    
        list of numpy arrays
            List of numpy arrays with generated x data - an array for each split.
        list of numpy arrays
            List of numpy arrays with generated y data - an array for each split.
        tiles : list of tuples
            List of tile coordinates.
        samples : list
            List with order of samples.
            
    Examples
    --------
        >>> x = ['/home/rsp_test/mosaics/sentinel/B1.tif',
        ... '/home/rsp_test/mosaics/sentinel/B2.tif',
        ... '/home/rsp_test/mosaics/sentinel/B3.tif',
        ... '/home/rsp_test/mosaics/sentinel/B4.tif',
        ... '/home/rsp_test/mosaics/sentinel/B5.tif',
        ... '/home/rsp_test/mosaics/sentinel/B6.tif',
        ... '/home/rsp_test/mosaics/sentinel/B7.tif',
        ... '/home/rsp_test/mosaics/sentinel/B8.tif',
        ... '/home/rsp_test/mosaics/sentinel/B8A.tif',
        ... '/home/rsp_test/mosaics/sentinel/B9.tif',
        ... '/home/rsp_test/mosaics/sentinel/B11.tif',
        ... '/home/rsp_test/mosaics/sentinel/B12.tif']
        >>> y = '/home/rsp_test/mosaics/landcover.tif'
        >>> s_file = '/home/rsp_test/model/samples.pickle'
        >>> x_train_file = '/home/rsp_test/model/x_train.npy'
        >>> x_val_file = '/home/rsp_test/model/x_val.npy'
        >>> x_test_file = '/home/rsp_test/model/x_test.npy'
        >>> y_train_file = '/home/rsp_test/model/y_train.npy'
        >>> y_val_file = '/home/rsp_test/model/y_val.npy'
        >>> y_test_file = '/home/rsp_test/model/y_test.npy'
        >>> x_i, y_i, tiles, samples = rsp.generate_tiles(x, y, num_classes = 11, tile_size = 256, shuffle = True, samples_file = s_file, split = [3, 1, 1], x_outputs = [x_train_file, x_val_file, x_test_file], y_outputs = [y_train_file, y_val_file, y_test_file], nodata = -1)
        >>> x_train = x_i[0]
        >>> print(x_train.shape)
        (3000, 256, 256, 12)
        >>> x_val = x_i[1]
        >>> print(x_val.shape)
        (1000, 256, 256, 12)
        >>> x_test = x_i[2]
        >>> print(x_test.shape)
        (1000, 256, 256, 12)
        >>> y_train = y_i[0]
        >>> print(y_train.shape)
        (3000, 256, 256, 11)
        >>> y_val = y_i[1]
        >>> print(y_val.shape)
        (1000, 256, 256, 11)
        >>> y_test = y_i[2]
        >>> print(y_test.shape)
        (1000, 256, 256, 11)
        >>> print(len(tiles))
        5000
        >>> print(samples[:5])
        [1876, 684, 25, 7916, 1347]
    """
    x, y, tiles, samples = get_tiles(x = x, y = y, tile_size = tile_size, num_classes = num_classes, categorical = categorical, shuffle = shuffle, samples_file = samples_file, split = split, x_outputs = x_outputs, y_outputs = y_outputs, dtype = dtype, nodata = nodata)
    if len(split) == 1:
        x = x[0]
        y = y[0]
    return x, y, tiles, samples
    
 
def generate_map(x, y_true, model, output, tiles = None, samples = None, samples_file = None, categorical = True, nodata = None):
    """
    Create map using pre-trained model.
    
    Parameters
    ----------
    x : numpy array with x data or path to .npy file with x data or list of arrays or paths 
        X tiled data that will be used for predictions. Usually it is data generated in `generate_tiles` function.
    y : path as a string
        Raster with target values which will be used as a reference raster to get size, transform and crs for a map.
    model : keras model or a path to a keras model
        Pre-trained model to predict target values.
    output : path as a string
        Path where to write output map
    tiles : list (optional)
        List of tile coordinates. Usually is generated in `generate_tiles` function. You also can use `samples_file` instead of `tiles` and `samples`.
    samples : list (optional) 
        List with order of samples. Usually is generated in `generate_tiles` function. You also can use `samples_file` instead of `tiles` and `samples`.
    samples_file : path as a string (optional)
        Path to a samples .pickle file generated by `generate_tiles` function. You can use `samples_file` instead of `tiles` and `samples`.
    categorical : bool (default = True)
        If y data is categorical. Usually True for classification and segmentation tasks and False for regression tasks.
    nodata : int or float (optional)
        Nodata value.
    
    Examples
    --------
        >>> x_i, y_i, tiles, samples = rsp.generate_tiles(x, y, num_classes = 11, tile_size = 256, shuffle = True, split = [3, 1, 1], nodata = -1)
        >>> x_train = x_i[0]
        >>> x_val = x_i[1]
        >>> x_test = x_i[2]
        >>> # Here model is initialised
        >>> model.fit(x_train, y_train, batch_size = 16, epochs = 20, validation_data = (x_val, y_val), callbacks = callbacks)
        >>> y_reference = '/home/rsp_test/mosaics/landcover.tif'
        >>> output_map = '/home/rsp_test/prediction.tif'
        >>> rsp.generate_map([x_train, x_val, x_test], y_reference, model, output_map, tiles = tiles, samples = samples, nodata = -1)
        
        >>> x_train_file = '/home/rsp_test/model/x_train.npy'
        >>> x_val_file = '/home/rsp_test/model/x_val.npy'
        >>> x_test_file = '/home/rsp_test/model/x_test.npy'
        >>> s_file = '/home/rsp_test/model/samples.pickle'
        >>> model = '/home/rsp_test/model/u-net.hdf5'
        >>> y_reference = '/home/rsp_test/mosaics/landcover.tif'
        >>> output_map = '/home/rsp_test/prediction.tif'
        >>> rsp.generate_map([x_train_file, x_val_file, x_test_file], y_reference, model, output_map, samples_file = s_file, nodata = -1)
    """
    if (tiles != None and samples != None) or (samples_file != None):
        if isinstance(x, list):
            x = [x]
        predict_map(x = x, y_true = y_true, model = model, categorical = categorical, samples_file = samples_file, tiles = tiles, samples = samples, output = output, nodata = nodata)
    else:
        print('Tiles and samples must be specified')