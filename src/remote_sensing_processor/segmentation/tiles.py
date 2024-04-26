import os
import warnings

import numpy as np
import xarray

import rioxarray

from remote_sensing_processor.common.common_functions import persist


def get_ss_tiles(
    x, 
    y, 
    tile_size, 
    classification, 
    shuffle, 
    split, 
    split_names, 
    x_output, 
    y_output, 
    x_dtype, 
    y_dtype, 
    x_nodata, 
    y_nodata
):
    x_names = [os.path.basename(i).split('.')[0] for i in x]
    if y != None:
        y_names = [os.path.basename(i).split('.')[0] for i in y]
    if x_nodata == None:
        x_nodata = get_nodata(x)
    # Calculate border and tile_size
    border = int(round(tile_size * 0.0625))
    tile_size = tile_size - (border * 2)
    # Reading all x files
    files = []
    for i in x:
        with rioxarray.open_rasterio(i, chunks=True, lock=True) as tif:
            band = persist(tif)
            if band.rio.nodata != x_nodata:
                warnings.warn(str(i) + " nodata value is " + str(band.rio.nodata)
                              + ". It will be converted to " + str(x_nodata))
                band = band.where(band.rio.nodata, x_nodata)
            files.append(band)
    x_img = xarray.concat(files, dim=xarray.Variable('band', x_names))
    if x_dtype != None:
        x_img = x_img.astype(x_dtype)
    x_img = persist(x_img)
    # Reading y files
    if y != None:
        files = []
        for i in y:
            with rioxarray.open_rasterio(i, chunks=True, lock=True) as tif:
                band = persist(tif)
                files.append(band)
        y_img = xarray.concat(files, dim=xarray.Variable('band', y_names))
        if y_dtype != None:
            y_img = y_img.astype(y_dtype)
        if y_nodata == None:
            y_nodata = y_img.rio.nodata
        # Checking image shapes
        assert x_img.shape[1:] == y_img.shape[1:]
        y_img = persist(y_img)
    # Calculate padding
    shp_in = x_img.shape[1:3]
    shp_pad = []
    for d in shp_in:
        while not d % tile_size == 0:
            d += 1
        shp_pad.append(d)
    # Padding
    x_img = x_img.pad(
        {'y': (0, shp_pad[0] - x_img.shape[1]), 'x': (0, shp_pad[1] - x_img.shape[2])}, 
        mode='constant', 
        constant_values=x_nodata
    )
    x_img = persist(x_img)
    if y != None:
        y_img = y_img.pad(
            {'y': (0, shp_pad[0] - y_img.shape[1]), 'x': (0, shp_pad[1] - y_img.shape[2])}, 
            mode='constant', 
            constant_values=y_nodata
        )
        y_img = persist(y_img)
    # Generating tiles
    tiles = []
    x1 = 0
    y1 = 0
    x2 = tile_size
    y2 = tile_size
    for i in range(0, int(shp_pad[0] / tile_size), 1):
        for i in range(0, int(shp_pad[1] / tile_size), 1):
            tiles.append((x1, y1, x2, y2))
            y1 += tile_size
            y2 += tile_size
        y1 = 0
        y2 = tile_size
        x1 += tile_size
        x2 += tile_size
    # Getting samples
    samples = list(range(len(tiles)))
    # Deleting tiles that contain only nodata (if needed)
    samples_filtered = []
    for i in samples:
        x_tile = x_img.isel(
            y=slice(tiles[i][0], tiles[i][2]), 
            x=slice(tiles[i][1], tiles[i][3])
        )
        if x_tile.mean().values.item() == x_nodata:
            if y != None:
                y_tile = y_img.isel(
                    y=slice(tiles[i][0], tiles[i][2]), 
                    x=slice(tiles[i][1], tiles[i][3])
                )
                if y_tile.mean().values.item() != y_nodata:
                    samples_filtered.append(i)
        else:
            samples_filtered.append(i)
    samples = samples_filtered
    # Shuffling samples
    if shuffle == True:
        np.random.shuffle(samples)
    # Splitting data
    split_samples = []
    j = 0
    for i in range(len(split)):
        split_samples.append(samples[j : j + round(split[i] / sum(split) * len(samples))])
        j = j + round(split[i] / sum(split) * len(samples))
    # Cutting x data into chips and stacking
    stack = [x_img.isel(y=slice(tiles[i][0], tiles[i][2]), x=slice(tiles[i][1], tiles[i][3])) for i in samples]
    x_img = xarray.concat(stack, 'chips', join="override").chunk('auto')
    x_img = persist(x_img)
    # Adding border to tiles to avoid border effects on predict
    x_img = x_img.pad({'y': border, 'x': border}, mode="symmetric").chunk('auto')
    x_img = persist(x_img)
    # Saving x data
    x_img = x_img.assign_attrs(
        tiles=tiles, 
        border=border, 
        samples=split_samples, 
        names=split_names
    )
    x_img.rio.write_nodata(x_nodata, inplace=True)
    if x_output != None:
        x_img.to_zarr(x_output, mode='w')
    # Cutting y data into chips and stacking
    y_data = []
    if y != None:
        for i in range(len(y_img)):
            y_var = y_img[i]
            if classification:
                classes = sorted(np.unique(y_var))
                num_classes = len(classes)
                y_nodata = classes.index(y_nodata)
                # Changing y values to range 0:n
                if not set([k == v for v, k in enumerate(classes)]) == {True}:
                    for v, k in enumerate(classes).items():
                        y_var = xr.where(y_var == k, v + 100000, y_var)
                    for v, _ in enumerate(classes).items():
                        y_var = xr.where(y_var == v + 100000, y_var - 100000, y_var)
                    #y_var = xarray.apply_ufunc(remap, y_var, classes, dask = 'allowed')
            else:
                num_classes = 1
                classes = 0
            # Cutting data into chips and stacking
            stack = [y_var.isel(y=slice(tiles[i][0], tiles[i][2]), x=slice(tiles[i][1], tiles[i][3])) for i in samples]
            y_var = xarray.concat(stack, 'chips', join="override").chunk('auto')
            y_var = persist(y_var)
            # Adding border to tiles to avoid border effects on predict
            y_var = y_var.pad({'y': border, 'x': border}, mode="symmetric").chunk('auto')
            y_var = persist(y_var)
            # Saving x data
            y_var = y_var.assign_attrs(
                tiles=tiles, 
                border=border, 
                samples=split_samples, 
                names=split_names, 
                classification=classification, 
                classes=classes, 
                num_classes=num_classes
            )
            y_var.rio.write_nodata(y_nodata, inplace=True)
            if y_output != None:
                y_var.to_zarr(y_output[i], mode='w')
            y_data.append(y_var)
    else:
        y_data = None
    return x_img, y_data


"""
def remap(array, classes):
    return np.vectorize({k: v for v, k in enumerate(classes)}.get)(array)
"""
    
    
def get_nodata(x):
    nodatas = []
    for i in x:
        with rioxarray.open_rasterio(i, chunks=True, lock=True) as f:
            nodatas.append(f.rio.nodata)
    nodata = max(set(nodatas), key=nodatas.count)
    return nodata