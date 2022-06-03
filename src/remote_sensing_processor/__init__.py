from glob import glob
import shutil

from remote_sensing_processor.unzip.unzip import unzip_sentinel, unzip_landsat

from remote_sensing_processor.sentinel2.sen2cor.sen2cor import sen2correct
from remote_sensing_processor.sentinel2.superres.superres import Superresolution
from remote_sensing_processor.sentinel2.sentinel_postprocessing.sentinel_postprocessing import s2postprocess_superres, s2postprocess_no_superres, get_first_proj

from remote_sensing_processor.landsat.landsat import landsat_proc

from remote_sensing_processor.mosaic.mosaic import mosaic_main, order, ismultiband

from remote_sensing_processor.indices.normalized_difference import nd
from remote_sensing_processor.imagery_types.types import get_type, get_index

from remote_sensing_processor.postprocessing.tiles import get_tiles, predict_map

def sentinel2(archives, sen2cor = True, superres = True, projection = None, cloud_mask = True):
    paths = []
    for archive in archives:
        path = unzip_sentinel(archive)
        path1 = glob(path+'*')[0]
        if sen2cor == True:
            sen2correct(path1)
        path1 = glob(path+'*')[0]
        if superres == True:
            Superresolution(input_dir = path1, output_dir = path1, copy_original_bands = True, clip_to_aoi = False, geometry = None, bounds = None).start()
            img = glob(path+'**\\*_superresolution.tif')[0]
            if projection == 'same':
                projection = get_first_proj(img)
            s2postprocess_superres(img = img, projection = projection, cloud_mask = cloud_mask, path = path, path1 = path1)
        else:
            if projection == 'same':
                img = glob(path1 + '\**\*.jp2')[0]
                projection = get_first_proj(img)
            s2postprocess_no_superres(projection = projection, cloud_mask = cloud_mask, path = path, path1 = path1)
        shutil.rmtree(path1)
        paths.append(path)
        print('Preprocessing of ' + archive + ' completed')
    return paths
    

def landsat(archives, projection = None, cloud_mask = True, pansharpen = True, keep_pan_band = False, resample = 'bilinear', t = 'k'):
    paths = []
    for archive in archives:
        path = unzip_landsat(archive)
        landsat_proc(path = path, projection = projection, cloud_mask = cloud_mask, pansharpen = pansharpen, keep_pan_band = keep_pan_band, resample = resample, t = t)
        paths.append(path)
        print('Preprocessing of ' + archive + ' completed')
    return paths


def mosaic(inputs, output_dir, fill_nodata = False, fill_distance = 250, clipper = None, crs = None, nodata = 0, reference_raster = None, nodata_order = False, keep_all_channels = True):
    mb = ismultiband(inputs[0])
    if mb == True:
        for i in range(len(inputs)):
            if not inputs[i].endswith(r'\\'):
                inputs[i] = inputs[i] + r'\\'
    if not output_dir.endswith(r'\\'):
        output_dir = output_dir + r'\\'
    if nodata_order == True:
        inputs = order(inputs)
    mosaic_main(inputs = inputs, output_dir = output_dir, fill_nodata = fill_nodata, fill_distance = fill_distance, clipper = clipper, crs = crs, nodata = nodata, reference_raster = reference_raster, mb = mb, keep_all_channels = keep_all_channels)


def normalized_difference(name, folder = None, b1 = None, b2 = None):
    if folder != None:
        if not folder.endswith(r'\\'):
            folder = folder + r'\\'
        t = get_type(folder)
        try:
            if t == 'Undefined':
                raise ValueError('Cannot define imagery type')
        except ValueError as e:
            print(e)
        b1, b2 = get_index(t, name, folder)
    try:
        if (b1 != None) and (b2 != None):
            nd(name = name, b1 = b1, b2 = b2, folder = folder)
        else:
            raise ValueError('Bands 1 and 2 must be defined')
    except ValueError as e:
        print(e)
        
             
def generate_tiles(x, y, num_classes, tile_size = 128, categorical = True, shuffle = False, samples_file = None, split = [1], x_outputs = None, y_outputs = None, dtype = None, nodata = None):
    x, y, tiles, samples = get_tiles(x = x, y = y, tile_size = tile_size, num_classes = num_classes, categorical = categorical, shuffle = shuffle, samples_file = samples_file, split = split, x_outputs = x_outputs, y_outputs = y_outputs, dtype = dtype, nodata = nodata)
    if len(split) == 1:
        x = x[0]
        y = y[0]
    return x, y, tiles, samples
    
 
def generate_map(x, y_true, model, output, tiles = None, samples = None, samples_file = None, categorical = True, nodata = None):
    if (tiles != None and samples != None) or (samples_file != None):
        predict_map(x = x, y_true = y_true, model = model, categorical = categorical, samples_file = samples_file, tiles = tiles, samples = samples, output = output, nodata = nodata)
    else:
        print('Tiles and samples must be specified')