import numpy as np
import h5py

import rasterio as rio

import tensorflow as tf
from tensorflow import keras


def get_tiles(x, y, tile_size, num_classes, categorical, shuffle, samples_file, split, x_outputs, y_outputs, dtype, nodata):
    shp_in, shp_work, shp_pad, xdtype, transform, crs = shapes(x[0], tile_size)
    if dtype == None:
        dtype = xdtype
    #creating np array
    x_train_conv = np.empty((shp_work[0], shp_work[1], len(x)), dtype = dtype)
    #reading all x files
    for i in range(len(x)):
        with rio.open(x[i]) as bnd:
            container = x_train_conv[:,:,i:i+1]
            container[:] = np.transpose(np.pad(bnd.read(), ((0,0),(0,shp_pad[0]),(0,shp_pad[1])), mode='constant', constant_values=0), axes=[1, 2, 0]).astype('uint16')
            #print(files[i])
    #reading y file
    with rio.open(y) as bnd:
        y_train_conv = bnd.read().astype(dtype)
    y_train_conv = np.pad(y_train_conv, ((0,0),(0,shp_pad[0]),(0,shp_pad[1])), mode='constant', constant_values=0)
    y_train_conv = np.transpose(y_train_conv, axes=[1, 2, 0])
    #generating tiles
    tiles = []
    x1 = 0
    y1 = 0
    x2 = tile_size
    y2 = tile_size
    for i in range (0,int(shp_work[0] / tile_size),1):
        for i in range(0,int(shp_work[1] / tile_size),1):
            tiles.append((x1,y1,x2,y2))
            y1+=tile_size
            y2+=tile_size
        y1=0
        y2=tile_size
        x1+=tile_size
        x2+=tile_size
    #shuffling samples
    samples = list(range(len(tiles)))
    if shuffle == True:
        np.random.shuffle(samples)
    #generating x tiles
    x_tiles = np.empty((len(samples), tile_size, tile_size, len(x)), dtype = dtype)
    for i in range(len(samples)):
        c = tiles[samples[i]]
        container = x_tiles[i:i+1,:,:,:]
        container[:] = x_train_conv[c[0]:c[2], c[1]:c[3],:]
    #generating y tiles
    y_tiles = np.empty((len(samples), tile_size, tile_size, 1), dtype = dtype)
    for i in range(len(samples)):
        c = tiles[samples[i]]
        container = y_tiles[i:i+1,:,:,:]
        container[:] = y_train_conv[c[0]:c[2], c[1]:c[3],:]
    #deleting tiles that contain only nodata (if needed)
    if nodata != None:
        todelete = []
        for i in range(len(y_tiles)):
            if set(np.unique(y_tiles[i])) == {nodata}:
                todelete.append(i)
        x_tiles = np.delete(x_tiles, todelete, axis = 0)
        y_tiles = np.delete(y_tiles, todelete, axis = 0)
        for index in sorted(todelete, reverse=True):
            del samples[index]
    if categorical == True:
        y_tiles = tf.keras.utils.to_categorical(y_tiles, num_classes=num_classes, dtype = 'float32')
    #splitting data
    if len(split) > 1:
        x_bags = []
        y_bags = []
        j = 0
        for i in range(len(split)):
            #generating samples subset
            x_bag = x_tiles[j:j+round(split[i]/sum(split)*len(samples))]
            y_bag = y_tiles[j:j+round(split[i]/sum(split)*len(samples))]
            #saving outputs
            if x_outputs != None:
                #np.savez(x_outputs[i], x_bag)
                with h5py.File(x_outputs[i], "w") as hf:
                    dset = f.create_dataset("data", data=x_bag, compression="gzip", compression_opts=9, shuffle=False)
            x_bags.append(x_bag)
            if y_outputs != None:
                #np.savez(y_outputs[i], y_bag)
                with h5py.File(y_outputs[i], "w") as hf:
                    dset = f.create_dataset("data", data=y_bag, compression="gzip", compression_opts=9, shuffle=False)
            y_bags.append(y_bag)
            #calculating start point for next bag
            j=j+round(split[i]/sum(split)*len(samples))
    else:
        x_bags = x_tiles
        y_bags = y_tiles
    if samples_file != None:
        with open(samples_file, "wb") as fp:
            pickle.dump([tiles, samples], fp)
    return x_bags, y_bags, tiles, samples
    
    
def predict_map_from_tiles(x, y_true, model, categorical, tiles, samples, samples_file, output, nodata):
    #loading tiles
    if samples_file != None:
        with open(samples_file, "rb") as fp:
            tiles, samples = pickle.load(fp)
    tile_size = tiles[0][2] - tiles[0][0]
    shp_in, shp_work, shp_pad, dtype, transform, crs = shapes(y_true, tile_size)
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)
    #creating map array
    if nodata == None:
        y_pred = np.zeros(shp_work, dtype)
    else:
        y_pred = np.full(shp_work, nodata, dtype)
    #mapping
    j = 0
    for i in range(len(x)):
        if isinstance(x[i], str):
            db = h5py.File(x[i])
            x_bag = db['data'][...]
            #x_bag = np.load(x[i])
        else:
            x_bag = x[i]
        bag = samples[j:len(x_bag)]
        j=len(x_bag)
        for i in range(len(bag)):
            #prediction for each tile
            prediction = model.predict(x_bag[i:i+1])
            if categorical == True:
                prediction = np.squeeze(np.argmax(prediction, axis=-1))
            #writing predicted tile to its position in array
            t = bag[i]
            t = tiles[t]
            y_pred[t[0]:t[2],t[1]:t[3]] = prediction
        if isinstance(x[i], str):
            db.close()
    #clipping
    y_pred = y_pred[:shp_in[0], :shp_in[1]]
    #writing to file
    with rio.open(
        output,
        'w',
        driver='GTiff',
        height=y_pred.shape[0],
        width=y_pred.shape[1],
        count=1,
        dtype=y_pred.dtype,
        compress = 'deflate',
        PREDICTOR = 1,
        ZLEVEL=9,
        crs=crs,
        transform=transform,
        nodata = nodata
    ) as outfile:
        outfile.write(y_pred, 1)
        
        
        
def shapes(i, tile_size):
    #getting shape, proj and transform of input image
    with rio.open(i) as f:
        dtype = f.dtypes[0]
        shp_in = f.shape
        transform = f.transform
        crs = f.crs
    #getting shape that is needed to divide into tiles without remainder
    shp_work = []
    for d in shp_in:
        while not d % tile_size == 0:
            d += 1
        shp_work.append(d)
    #getting how much rows and cols we need to add to input shape
    shp_pad = []
    for i in range(len(shp_work)):
        a = shp_work[i]
        b = shp_in[i]
        shp_pad.append(a - b)
    return shp_in, shp_work, shp_pad, dtype, transform, crs
        
        