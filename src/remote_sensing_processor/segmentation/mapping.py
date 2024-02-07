import joblib

import numpy as np
import xarray

import rioxarray

import torch
import lightning as l

from remote_sensing_processor.common.common_functions import persist

from remote_sensing_processor.segmentation.segmentation import Model, SegDataModule


def predict_map_from_tiles(x, y, reference, model, output, nodata, batch_size, num_workers):
    # Loading reference raster
    with rioxarray.open_rasterio(reference, chunks = True, lock = True) as tif:
        reference = persist(tif)
    if isinstance(model, str):
        if '.ckpt' in model:
            model = Model.load_from_checkpoint(model)
        elif '.joblib' in model:
            model = joblib.load(model)
    # Reading classes and nodata
    if isinstance(y, str):
        y_dataset = xarray.open_dataarray(y, engine = 'zarr', chunks = 'auto', mask_and_scale = False)
    elif isinstance(y, xarray.DataArray):
        y_dataset = y
    if nodata == None:
        nodata = y_dataset.rio.nodata
    classes = y_dataset.classes
    # Creating empty array
    y = xarray.full_like(reference[0], nodata)
    # Neural networks
    if model.model_name in ['BEiT', 'ConditionalDETR', 'Data2Vec', 'DETR', 'DPT', 'Mask2Former', 'MaskFormer', 'MobileNetV2', 'MobileViT', 'MobileViTV2', 'OneFormer', 'SegFormer', 'UperNet', 'DeepLabV3', 'FCN', 'LRASPP']:
        # Setting datamodule
        dm = SegDataModule(pred_dataset = x, batch_size = batch_size, num_workers = num_workers)
        dm.setup(stage = 'predict')
        tiles = dm.ds_pred.tiles
        border = dm.ds_pred.border
        samples = [x for x in dm.ds_pred.samples]
        # Predict
        trainer = l.Trainer()
        predictions = trainer.predict(model, dm)
    # Sklearn models
    elif model.model_name in ["Nearest Neighbors", "Logistic Regression", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Gradient Boosting", "Multilayer Perceptron", "AdaBoost", "Naive Bayes", "QDA", "Ridge", "Lasso", "ElasticNet", "XGBoost", "XGB Random Forest"]:
        if isinstance(x, str):
            x_dataset = xarray.open_dataarray(x, engine = 'zarr', chunks = 'auto', mask_and_scale = False)
        else:
            x_dataset = x
        x_dataset = persist(x_dataset)
        tiles = x_dataset.tiles
        border = x_dataset.border
        samples = [x for xs in x_dataset.samples for x in xs]
        input_shape = x_dataset.shape[2]
        # Predict every tile
        predictions = []
        for i in x_dataset:
            i = i.astype('float32').stack(data = ('y', 'x')).transpose('data', 'band')
            prediction = model.predict(i)
            prediction = prediction.reshape(input_shape, input_shape)
            predictions.append(prediction[np.newaxis, :])
    else:
        raise ValueError("Wrong model name. Check spelling or read a documentation and choose a supported model")
    predictions = np.concatenate(predictions, axis = 0)
    # Mapping
    for i in range(len(samples)):
        prediction = predictions[i]
        # Writing predicted tile to its position in array
        t = tiles[samples[i]]
        # Cutting off border and cutting the last prediction
        if t[2] > y.shape[0] or t[3] > y.shape[1]:
            t[0], t[1], t[2], t[3] = (t[0], t[1], min(t[2], y.shape[0]), min(t[3], y.shape[1]))
            prediction = prediction[border : t[2] - t[0] + border, border : t[3] - t[1] + border]
        else:
            prediction = prediction[border : prediction.shape[0] - border, border : prediction.shape[1] - border]
        # TODO: not working with dask arrays, so had to load reference without chunking, can be memory-consuming.
        area = y.isel(y = slice(t[0], t[2]), x = slice(t[1], t[3]))
        y.loc[{'x': area.x, 'y': area.y}] = prediction
    # Recreating original classes values
    if classes != 0 and not set([k == v for v, k in enumerate(classes)]) == {True}:
        for k, v in enumerate(classes).items():
            y = xr.where(y == k, v + 100000, y)
        for _, v in enumerate(classes).items():
            y = xr.where(y == v + 100000, y - 100000, y)
    y = persist(y)
    # Because predictor = 2 works with float64 only when libtiff > 3.2.0 is installed and default libtiff in ubuntu is 3.2.0
    if y.dtype == 'float64':
        y = y.astype('float32')
        y = persist(y)
    # Writing to file
    y.rio.to_raster(output, compress = 'deflate', PREDICTOR = 2, ZLEVEL = 9, BIGTIFF = 'IF_SAFER', tiled = True, NUM_THREADS = 'NUM_CPUS', lock = True)