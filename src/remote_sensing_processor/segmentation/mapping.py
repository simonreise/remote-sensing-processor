import numpy as np
import pickle

import rasterio as rio

import torch
import lightning as l

from remote_sensing_processor.segmentation.tiles import shapes
from remote_sensing_processor.segmentation.segmentation import Model, PredDataset


def predict_map_from_tiles(x, y_true, model, tiles, samples, classes, samples_file, output, nodata, batch_size):
    #loading tiles
    if samples_file != None:
        with open(samples_file, "rb") as fp:
            tiles, samples, classes = pickle.load(fp)
    tile_size = tiles[0][2] - tiles[0][0]
    shp_in, shp_work, shp_pad, dtype, transform, crs, y_nodata = shapes(y_true, tile_size)
    if nodata == None:
        nodata = y_nodata
    if isinstance(model, str):
        model = Model.load_from_checkpoint(model)
    #creating map array
    if nodata == None:
        y_pred = np.zeros(shp_work, dtype)
    else:
        y_pred = np.full(shp_work, nodata, dtype)
    #setting data loaders
    datasets = []
    for i in range(len(x)):
        datasets.append(PredDataset(x[i]))
    ds = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True)
    #prediction    
    trainer = l.Trainer()
    predictions = trainer.predict(model, dataloaders=loader)
    predictions = np.concatenate(predictions, axis=0)
    #mapping
    for i in range(len(samples)):
        prediction = predictions[i]
        #writing predicted tile to its position in array
        t = samples[i]
        t = tiles[t]
        y_pred[t[0]:t[2],t[1]:t[3]] = prediction
    #clipping
    y_pred = y_pred[:shp_in[0], :shp_in[1]]
    #recreating original classes values
    if classes != None:
        y_pred = np.vectorize({v: k for v, k in enumerate(classes)}.get)(y_pred)
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