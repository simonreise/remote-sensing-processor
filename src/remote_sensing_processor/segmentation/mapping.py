import numpy as np
import pickle
import joblib
import h5py

import rasterio as rio

import torch
import lightning as l

from remote_sensing_processor.segmentation.tiles import shapes
from remote_sensing_processor.segmentation.segmentation import Model, PredDataset


def predict_map_from_tiles(x, y_true, model, tiles, samples, classes, samples_file, output, nodata, batch_size, multiprocessing):
    #loading tiles
    if samples_file != None:
        with open(samples_file, "rb") as fp:
            tiles, samples, classes = pickle.load(fp)
    tile_size = tiles[0][2] - tiles[0][0]
    shp_in, shp_work, shp_pad, dtype, transform, crs, y_nodata = shapes(y_true, tile_size)
    if nodata == None:
        nodata = y_nodata
    if isinstance(model, str):
        if '.ckpt' in model:
            model = Model.load_from_checkpoint(model)
        elif '.joblib' in model:
            model = joblib.load(model)
    #creating map array
    if nodata == None:
        y_pred = np.zeros(shp_work, dtype)
    else:
        y_pred = np.full(shp_work, nodata, dtype)
    if model.model_name in ['BEiT', 'ConditionalDETR', 'Data2Vec', 'DETR', 'DPT', 'Mask2Former', 'MaskFormer', 'MobileNetV2', 'MobileViT', 'MobileViTV2', 'OneFormer', 'SegFormer', 'UperNet', 'DeepLabV3', 'FCN', 'LRASPP']:
        #neural networks
        #setting data loaders
        datasets = []
        for i in range(len(x)):
            datasets.append(PredDataset(x[i]))
        ds = torch.utils.data.ConcatDataset(datasets)
        if multiprocessing:
            cpus = torch.multiprocessing.cpu_count()
        else:
            cpus = 0
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers = cpus)
        #prediction    
        trainer = l.Trainer()
        predictions = trainer.predict(model, dataloaders=loader)
    elif model.model_name in ["Nearest Neighbors", "Logistic Regression", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Gradient Boosting", "Multilayer Perceptron", "AdaBoost", "Naive Bayes", "QDA", "Ridge", "Lasso", "ElasticNet"]:
        #sklearn models
        datasets = []
        #concatenate all datasets
        for i in x:
            if isinstance(i, str):
                with h5py.File(i, 'r') as file:
                    ds = file['data']
                    ds = ds[...]
                    datasets.append(ds)
            else:
                datasets.append(i)
        datasets = np.concatenate(datasets, axis = 0)
        #predict every tile
        predictions = []
        #print(datasets.shape)
        for i in datasets:
            #print(i.shape)
            size = i.shape[1]
            i = i.reshape(i.shape[0], -1)
            #print(i.shape)
            i = np.moveaxis(i, 0, -1)
            #print(i.shape)
            prediction = model.predict(i)
            #print(prediction.shape)
            prediction = prediction.reshape(size, size)
            #print(prediction.shape)
            predictions.append(prediction[np.newaxis, :])
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
    if classes != 0:
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