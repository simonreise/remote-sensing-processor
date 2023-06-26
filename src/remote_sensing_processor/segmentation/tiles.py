import numpy as np
import h5py
import pickle

import rasterio as rio


def get_ss_tiles(x, y, tile_size, classification, shuffle, samples_file, split, x_outputs, y_outputs, x_dtype, y_dtype, x_nodata, y_nodata):
    shp_in, shp_work, shp_pad, xdtype, transform, crs, xnodata = shapes(x[0], tile_size)
    if x_nodata == None:
        x_nodata = xnodata
    if x_dtype == None:
        x_dtype = xdtype
    #creating np array
    #x_train_conv = np.empty((shp_work[0], shp_work[1], len(x)), dtype = x_dtype)
    x_train_conv = np.empty((len(x), shp_work[0], shp_work[1]), dtype = x_dtype)
    #reading all x files
    for i in range(len(x)):
        with rio.open(x[i]) as bnd:
            container = x_train_conv[i:i+1,:,:]
            container[:] = np.pad(bnd.read().astype(x_dtype), ((0,0),(0,shp_pad[0]),(0,shp_pad[1])), mode='constant', constant_values=0)
            #container = x_train_conv[:,:,i:i+1]
            #container[:] = np.transpose(np.pad(bnd.read().astype(x_dtype), ((0,0),(0,shp_pad[0]),(0,shp_pad[1])), mode='constant', constant_values=0), axes=[1, 2, 0])
            #print(files[i])
    #reading y file
    with rio.open(y) as bnd:
        if y_nodata == None:
            y_nodata = bnd.nodata
        if y_dtype == None:
            y_dtype = bnd.dtypes[0]
        y_train_conv = np.pad(bnd.read().astype(y_dtype), ((0,0),(0,shp_pad[0]),(0,shp_pad[1])), mode='constant', constant_values=0)
        #y_train_conv = np.transpose(np.pad(bnd.read().astype(y_dtype), ((0,0),(0,shp_pad[0]),(0,shp_pad[1])), mode='constant', constant_values=0), axes=[1, 2, 0])
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
    x_tiles = np.empty((len(samples), len(x), tile_size, tile_size), dtype = x_dtype)
    #x_tiles = np.empty((len(samples), tile_size, tile_size, len(x)), dtype = x_dtype)
    for i in range(len(samples)):
        c = tiles[samples[i]]
        container = x_tiles[i:i+1,:,:,:]
        container[:] = x_train_conv[:, c[0]:c[2], c[1]:c[3]]
        #container[:] = x_train_conv[c[0]:c[2], c[1]:c[3],:]
    #make class values 0, 1, 2...
    if classification:
        classes = sorted(np.unique(y_train_conv))
        num_classes = len(classes)
        y_nodata = classes.index(y_nodata)
        y_train_conv = np.vectorize({k: v for v, k in enumerate(classes)}.get)(y_train_conv)
        #num_classes = int(np.max(y_train_conv) + 1)
        #y_nodata = int(y_nodata)
    else:
        num_classes = 1
        classes = None
    #generating y tiles
    y_tiles = np.empty((len(samples), tile_size, tile_size), dtype = y_dtype)
    #y_tiles = np.empty((len(samples), tile_size, tile_size, 1), dtype = y_dtype)
    for i in range(len(samples)):
        c = tiles[samples[i]]
        container = y_tiles[i:i+1,:,:]
        #if categorical:
            #converting data to one hot
            #container[:] = np.transpose(np.eye(num_classes, dtype='uint8')[y_train_conv[0][c[0]:c[2], c[1]:c[3]]], axes = [2, 0, 1])
        #else:
        container[:] = y_train_conv[0][c[0]:c[2], c[1]:c[3]]
        #container[:] = y_train_conv[c[0]:c[2], c[1]:c[3],:]
    #deleting tiles that contain only nodata (if needed)
    if (x_nodata != None) or (y_nodata != None):
        #getting y nodata tiles
        #nodata_fixes_needed = False
        ytodelete = []
        if y_nodata != None:
            for i in range(len(y_tiles)):
                #if categorical:
                    #if set(np.unique(y_tiles[i][y_nodata])) == {1}:
                        #ytodelete.append(i)
                #else:
                if set(np.unique(y_tiles[i])) == {y_nodata}:
                    ytodelete.append(i)
                #check if there are areas where x nodata corresponds to y data and vice versa
                #if not np.array_equal(y_tiles[i] == y_nodata, x_tiles[i][:,:,0:1] == x_nodata):
                #if categorical:
                    #if not np.array_equal(y_tiles[i][y_nodata] == 1, x_tiles[i][0] == x_nodata):
                        #nodata_fixes_needed = True
                #else:
                    #if not np.array_equal(y_tiles[i][0] == y_nodata, x_tiles[i][0] == x_nodata):
                        #nodata_fixes_needed = True
        #getting x nodata tiles
        xtodelete = []
        if x_nodata != None:           
            for i in range(len(x_tiles)):
                if set(np.unique(x_tiles[i])) == {x_nodata}:
                    xtodelete.append(i)
        todelete = list(set(xtodelete) & set(ytodelete))
        x_tiles = np.delete(x_tiles, todelete, axis = 0)
        y_tiles = np.delete(y_tiles, todelete, axis = 0)
        for index in sorted(todelete, reverse=True):
            del samples[index]
    #if categorical == True:
        #y_nodata = np.where(np.unique(y_tiles) == y_nodata)[0][0]
        #y_tiles = to_categorical(y_tiles, num_classes=num_classes, dtype = y_dtype)
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
                    dset = hf.create_dataset("data", data=x_bag, compression="gzip", compression_opts=9, shuffle=False)
                    dset.attrs['nodata'] = x_nodata
            x_bags.append(x_bag)
            if y_outputs != None:
                #np.savez(y_outputs[i], y_bag)
                with h5py.File(y_outputs[i], "w") as hf:
                    dset = hf.create_dataset("data", data=y_bag, compression="gzip", compression_opts=9, shuffle=False)
                    dset.attrs['nodata'] = y_nodata
                    dset.attrs['classification'] = classification
                    dset.attrs['classes'] = classes
                    dset.attrs['num_classes'] = num_classes
                    #dset.attrs['nfn'] = nodata_fixes_needed
            y_bags.append(y_bag)
            #calculating start point for next bag
            j=j+round(split[i]/sum(split)*len(samples))
    else:
        x_bags = x_tiles
        y_bags = y_tiles
    if samples_file != None:
        with open(samples_file, "wb") as fp:
            pickle.dump([tiles, samples, classes], fp)
    return x_bags, y_bags, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata


def shapes(i, tile_size):
    #getting shape, proj and transform of input image
    with rio.open(i) as f:
        dtype = f.dtypes[0]
        shp_in = f.shape
        transform = f.transform
        crs = f.crs
        nodata = f.nodata
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
    return shp_in, shp_work, shp_pad, dtype, transform, crs, nodata
        
        