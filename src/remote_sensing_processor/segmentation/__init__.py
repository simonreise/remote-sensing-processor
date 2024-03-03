import os
import numpy as np

import xarray

from remote_sensing_processor.common.torch_test import cuda_test

from remote_sensing_processor.segmentation.segmentation import segmentation_train, segmentation_test
from remote_sensing_processor.segmentation.tiles import get_ss_tiles
from remote_sensing_processor.segmentation.mapping import predict_map_from_tiles


def generate_tiles(x, y, tile_size = 128, classification = True, shuffle = False, split = [1], split_names = ['train'], x_output = None, y_output = None, x_dtype = None, y_dtype = None, x_nodata = None, y_nodata = None):
    """
    Cut rasters into tiles.
    
    Parameters
    ----------
    x : list of paths as strings
        Rasters to use as training data.
    y : path as a string or list of paths as strings
        Raster or multiple rasters to use as target variable. Can be set to None if target value is not needed.
    tile_size : int (default = 128)
        Size of tiles to generate (tile_size x tile_size).
    classification : bool (default = True)
        If True then tiles will be prepared for classification (e.g. semantic segmentation) task, else will be prepared for regression task.
    shuffle : bool (default = False)
        Is random shuffling of samples needed.
    split : list of ints or floats (optional)
        Splitting data in subsets. Is a list of integers defining proportions of every subset. [3, 1, 1] will generate 3 subsets in proportion 3 to 1 to 1.
    split_names : list of strings
        Names of split subsets.
    x_output : path as a string (optional)
        Path to save generated output x data. Data is saved in .zarr format.
    y_output : path as a string or list of paths as strings (optional)
        Path or list of paths to save generated output y data. Data is saved in .zarr format.
    x_dtype : dtype definition as a string (optional)
        If you run out of memory, you can try to convert your data to less memory consuming format.
    y_dtype : dtype definition as a string (optional)
        If you run out of memory, you can try to convert your data to less memory consuming format.
    x_nodata : int or float (optional)
        You can define which value in x raster corresponds to nodata and areas that contain nodata in x raster will be ignored while training and testing. Tiles that contain only nodata in both x and y will be omited. If not defined then nodata of first x file will be used.
    y_nodata : int or float (optional)
        You can define which value in y raster corresponds to nodata and areas that contain nodata in y raster will be ignored while training and testing. Tiles that contain only nodata in both x and y will be omited. If not defined then nodata of y file will be used.
    
    Returns
    ----------
    tuple:
    
        xarray.Dataarray
            Array with generated x data.
        xarray.Dataarray or list of xarray.Dataarray or None
            List of arrays with generated y data - one array for each y raster.
            
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
        >>> y = ['/home/rsp_test/mosaics/landcover.tif', 
        ... '/home/rsp_test/mosaics/forest_types.tif']
        >>> x_file = '/home/rsp_test/model/x.zarr'
        >>> y_files = ['/home/rsp_test/model/y_landcover.zarr', 
        ... '/home/rsp_test/model/y_forest_types.zarr']
        >>> x_out, y_out = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1], split_names = ['train', 'val', 'test'], x_output = x_file, y_output = y_files, x_nodata = 0, y_nodata = 0)
        >>> print(x_out.shape)
        (12, 8704, 6912)
        >>> y_landcover = y_out[0]
        >>> print(y_landcover.shape)
        (8704, 6912)
        >>> y_forest_types = y_out[1]
        >>> print(y_forest_types.shape)
        (8704, 6912)
    """
    # Type checking
    if isinstance(x, str):
        x = [x]
    elif isinstance(x, list):
        for i in x:
            if not isinstance(i, str):
                raise TypeError("x must be a string or a list of strings")
    else:
        raise TypeError("x must be a string or a list of strings")
    for i in x:
        if not os.path.exists(i):
            raise OSError(i + " does not exist")
    if isinstance(y, str):
        y = [y]
    elif isinstance(y, list):
        for i in y:
            if not isinstance(i, str):
                raise TypeError("y must be a string or a list of strings")
    elif not isinstance(y, type(None)):
        raise TypeError("y must be a string or a list of strings")
    if isinstance(y, list):
        for i in y:
            if not os.path.exists(i):
                raise OSError(i + " does not exist")
    if not isinstance(tile_size, int):
        if isinstance(tile_size, type(None)):
            tile_size = 128
        else:
            raise TypeError("tile_size must be an integer")
    else:
        if tile_size <= 8:
            raise ValueError("tile_size must be > 8")
    if not isinstance(classification, bool):
        if isinstance(classification, type(None)):
            classification = True
        else:
            raise TypeError("classification must be boolean")
    if not isinstance(shuffle, bool):
        if isinstance(shuffle, type(None)):
            shuffle = False
        else:
            raise TypeError("shuffle must be boolean")
    if isinstance(split, list):
        for i in split:
            if not isinstance(i, int) and not isinstance(i, float):
                raise TypeError("split must be a list of ints or floats")
    elif isinstance(split, type(None)):
        split = [1]
    else:
        raise TypeError("split must be a list of ints or floats")
    if isinstance(split_names, list):
        for i in split_names:
            if not isinstance(i, str):
                raise TypeError("split_names must be a list of strings")
    elif isinstance(split_names, type(None)):
        split_names = ['train']
    else:
        raise TypeError("split_names must be a list of strings")
    assert len(split) == len(split_names)
    if not isinstance(x_output, str) and not isinstance(x_output, type(None)):
        raise TypeError("x_outputs must be a string")
    if isinstance(y_output, str):
        y_output = [y_output]
    if isinstance(y_output, list):
        assert len(y_output) == len(y)
        for i in y_output:
            if not isinstance(i, str):
                raise TypeError("y_output must be a string or a list of strings")
    elif not isinstance(y_output, type(None)):
        raise TypeError("y_output must be a list of strings")
    if not isinstance(x_dtype, type(None)):
        np.dtype(x_dtype)
    if not isinstance(y_dtype, type(None)):
        np.dtype(y_dtype)
    if not isinstance(x_nodata, int) and not isinstance(x_nodata, float) and not isinstance(x_nodata, type(None)):
        raise TypeError("x_nodata must be integer or float")
    if not isinstance(y_nodata, int) and not isinstance(y_nodata, float) and not isinstance(y_nodata, type(None)):
        raise TypeError("y_nodata must be integer or float")
    
    x, y = get_ss_tiles(x = x, y = y, tile_size = tile_size, classification = classification, shuffle = shuffle, split = split, split_names = split_names, x_output = x_output, y_output = y_output, x_dtype = x_dtype, y_dtype = y_dtype, x_nodata = x_nodata, y_nodata = y_nodata)
    return x, y

    
def train(train_datasets, val_datasets, model_file, model, backbone = None, checkpoint = None, weights = None, epochs = 5, batch_size = 32, repeat = 1, augment = False, less_metrics = False, lr = 1e-3, num_workers = 0, classification = None, num_classes = None, y_nodata = None, **kwargs):
    """
    Trains segmentation model.
    
    Parameters
    ----------
    train_datasets : list or list of lists
        Training data generated by generate_tiles() function. Each dataset is a list of 3 elements: training data (x): file path or xarray.DataArray, target variable (y): file path or xarray.DataArray, split_names: string or list of strings or 'all' if you need to use the whole dataset. You can provide a list of datasets to train model on multiple datasets.
    val_datasets : list or list of lists or None
        Validation data generated by generate_tiles() function. Each dataset is a list of 3 elements: training data (x): file path or xarray.DataArray, target variable (y): file path or xarray.DataArray, split_names: string or list of strings or 'all' if you need to use the whole dataset. You can provide a list of datasets to validate model on multiple datasets. Can be set to None if no validation needed.
    model_file : path as a string
        Checkpoint file where model will be saved after training. File extension must be *.ckpt for neural networks and *.joblib for scikit-learn models.
    model : str
        Name of model architecture.
    backbone : str (optional)
        Backbone, solver or kernel of a model, if multiple backbones are supported.
    checkpoint : path as a string (optional)
        Checkpoint file (*.ckpt or *.joblib) of a pre-trained model to fine-tune.
    weights : str
        Name of pre-trained weights to fine-tune. Only works for neural networks.
    epochs : int (default = 5)
        Number of training epochs. Only works for neural networks and multilayer perceptron.
    batch_size : int (default = 32)
        Number of training samples used in one iteration. Only works for neural networks.
    repeat : int (default = 1)
        Increase size of a dataset by repeating it n times.
    augment : bool (default = False)
        Apply augmentations to dataset.
    less_metrics : bool (default = False)
        Sometimes Torchmetrics can freeze while calculating precision, recall and IOU. If it happens, try restarting with `less_metrics = True`.
    lr : float (default = 1e-3)
        Learning rate of a model. Lower value results usually in better model convergence, but much slower training.
    num_workers: int or 'auto' (default = 0)
        Number of parallel workers that will load the data. Set 'auto' to let RSP choose the optimal number of workers, set 0 to disable multiprocessing. Can increase training speed, but can also cause errors (e.g. pickling errors).
    classification : bool (default = None)
        If True then perform classification (e.g. semantic segmentation) task, else perform regression task. If not defined then is read from from train dataset.
    num_classes: int (optional)
        Number of classes for classification task. If not defined then is read from train dataset.
    y_nodata : int or float (optional)
        You can define which value in y raster corresponds to nodata and areas that contain nodata in y raster will be ignored while training and testing. If not defined then is read from train dataset.
    **kwargs
        Additional keyword arguments that are used to initialise model. They are different for every model, so read the documentation.
    
    Returns
    ----------
    torch.nn model or SklearnModel
        Trained model.
            
    Examples
    --------
        >>> x_out, y_out = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1], split_names = ['train', 'val', 'test'])
        >>> train_ds = [x_out, y_out[0], 'train']
        >>> val_ds = [x_out, y_out[0], 'val']
        >>> model = rsp.segmentation.train(train_ds, val_ds, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 100, batch_size = 32)
        GPU available: True (cuda), used: True
        TPU available: False, using: 0 TPU cores
        IPU available: False, using: 0 IPUs
        HPU available: False, using: 0 HPUs
        LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
          | Name    | Type                           | Params
        -----------------------------------------------------------
        0 | model   | UperNetForSemanticSegmentation | 59.8 M
        1 | loss_fn | CrossEntropyLoss               | 0     
        -----------------------------------------------------------
        59.8 M    Trainable params
        0         Non-trainable params
        59.8 M    Total params
        239.395   Total estimated model params size (MB)
        Epoch 9: 100% #############################################
        223/223 [1:56:20<00:00, 31.30s/it, v_num=54, train_loss_step=0.326, train_acc_step=0.871, train_auroc_step=0.796, train_iou_step=0.655,
        val_loss_step=0.324, val_acc_step=0.869, val_auroc_step=0.620, val_iou_step=0.678,
        val_loss_epoch=0.334, val_acc_epoch=0.807, val_auroc_epoch=0.795, val_iou_epoch=0.688,
        train_loss_epoch=0.349, train_acc_epoch=0.842, train_auroc_epoch=0.797, train_iou_epoch=0.648]
        `Trainer.fit` stopped: `max_epochs=10` reached.
        
        >>> x_mo = '/home/rsp_test/model/x_montana.zarr'
        >>> y_mo = '/home/rsp_test/model/y_montana.zarr'
        >>> x_id = '/home/rsp_test/model/x_idaho.zarr'
        >>> y_id = '/home/rsp_test/model/y_idaho.zarr'
        >>> train_datasets = [[x_mo, y_mo, ['area_1', 'area_2']], [x_id, y_id, ['area_3', 'area_6', 'area8']]]
        >>> val_datasets = [[x_mo, y_mo, ['area_3', 'area_4']], [x_id, y_id, ['area_1']]]
        >>> model = rsp.segmentation.train(train_datasets, val_datasets, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 100, batch_size = 32)
        GPU available: True (cuda), used: True
        TPU available: False, using: 0 TPU cores
        IPU available: False, using: 0 IPUs
        HPU available: False, using: 0 HPUs
        LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
          | Name    | Type                           | Params
        -----------------------------------------------------------
        0 | model   | UperNetForSemanticSegmentation | 59.8 M
        1 | loss_fn | CrossEntropyLoss               | 0     
        -----------------------------------------------------------
        59.8 M    Trainable params
        0         Non-trainable params
        59.8 M    Total params
        239.395   Total estimated model params size (MB)
        Epoch 99: 100% #############################################
        223/223 [1:56:20<00:00, 31.30s/it, v_num=54, train_loss_step=0.326, train_acc_step=0.871, train_auroc_step=0.796, train_iou_step=0.655,
        val_loss_step=0.324, val_acc_step=0.869, val_auroc_step=0.620, val_iou_step=0.678,
        val_loss_epoch=0.334, val_acc_epoch=0.807, val_auroc_epoch=0.795, val_iou_epoch=0.688,
        train_loss_epoch=0.349, train_acc_epoch=0.842, train_auroc_epoch=0.797, train_iou_epoch=0.648]
        `Trainer.fit` stopped: `max_epochs=100` reached.
    """
    # Type checking
    if not isinstance(train_datasets[0], list):
        train_datasets = [train_datasets]
    for i in range(len(train_datasets)):
        if len(train_datasets[i]) != 3:
            raise ValueError("Every dataset must consist of x, y and names")
        if not isinstance(train_datasets[i][0], str) and not isinstance(train_datasets[i][0], xarray.DataArray):
            raise TypeError("x in dataset must be a string or xarray.DataArray")
        elif isinstance(train_datasets[i][0], str) and not os.path.exists(train_datasets[i][0]):
            raise OSError(train_datasets[i] + " does not exist")
        if not isinstance(train_datasets[i][1], str) and not isinstance(train_datasets[i][1], xarray.DataArray):
            raise TypeError("y in dataset must be a string or xarray.DataArray")
        elif isinstance(train_datasets[i][1], str) and not os.path.exists(train_datasets[i][1]):
            raise OSError(train_datasets[i] + " does not exist")
        if not isinstance(train_datasets[i][2], str) and not isinstance(train_datasets[i][2], list):
            raise TypeError("name in dataset must be a string or a list")
        else:
            if train_datasets[i][2] != 'all' and isinstance(train_datasets[i][2], str):
                train_datasets[i][2] = [train_datasets[i][2]]
    if val_datasets != None:
        if not isinstance(val_datasets[0], list):
            val_datasets = [val_datasets]
        for i in val_datasets:
            if len(i) != 3:
                raise ValueError("Every dataset must consist of x, y and names")
            if not isinstance(i[0], str) and not isinstance(i[0], xarray.DataArray):
                raise TypeError("x in dataset must be a string or xarray.DataArray")
            elif isinstance(i[0], str) and not os.path.exists(i[0]):
                raise OSError(i + " does not exist")
            if not isinstance(i[1], str) and not isinstance(i[1], xarray.DataArray):
                raise TypeError("y in dataset must be a string or xarray.DataArray")
            elif isinstance(i[1], str) and not os.path.exists(i[1]):
                raise OSError(i + " does not exist")
            if not isinstance(i[2], str) and not isinstance(i[2], list):
                raise TypeError("name in dataset must be a string or a list")
    if not isinstance(model_file, str):
        raise TypeError("model_file must be a string")
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if not isinstance(backbone, str) and not isinstance(backbone, type(None)):
        raise TypeError("backbone must be a string")
    if not isinstance(checkpoint, str) and not isinstance(checkpoint, type(None)):
        raise TypeError("checkpoint must be a string")
    elif isinstance(checkpoint, str) and not os.path.exists(checkpoint):
        raise OSError(checkpoint + " does not exist")
    if not isinstance(weights, str) and not isinstance(weights, type(None)):
        raise TypeError("weights must be a string")
    if not isinstance(epochs, int):
        if isinstance(epochs, type(None)):
            epochs = 5
        else:
            raise TypeError("epochs must be an integer")
    if not isinstance(batch_size, int):
        if isinstance(batch_size, type(None)):
            batch_size = 32
        else:
            raise TypeError("batch_size must be an integer")
    if not isinstance(repeat, int):
        if isinstance(repeat, type(None)):
            repeat = 1
        else:
            raise TypeError("repeat must be an integer")
    elif repeat < 1:
        raise ValueError("repeat must be >= 1")
    if not isinstance(augment, bool):
        if isinstance(augment, type(None)):
            augment = False
        else:
            raise TypeError("augment must be boolean")
    if not isinstance(less_metrics, bool):
        if isinstance(less_metrics, type(None)):
            less_metrics = False
        else:
            raise TypeError("less_metrics must be boolean")
    if not isinstance(lr, float):
        if isinstance(lr, type(None)):
            lr = 1e-3
        else:
            raise TypeError("lr must be float")
    if (not isinstance(num_workers, int) and num_workers != 'auto') or (isinstance(num_workers, int) and num_workers < 0):
        if isinstance(num_workers, type(None)):
            num_workers = 'auto'
        else:
            raise TypeError("num_workers must be non-negative integer or 'auto'")
    if not isinstance(classification, bool) and not isinstance(classification, type(None)):
        raise TypeError("classification must be boolean or None")
    if not isinstance(num_classes, int) and not isinstance(num_classes, type(None)):
        raise TypeError("num_classes must be int or None")
    if not isinstance(y_nodata, int) and not isinstance(y_nodata, float) and not isinstance(y_nodata, type(None)):
        raise TypeError("y_nodata must be int or float or None")
    
    cuda = cuda_test()
    if cuda == False:
        warnings.warn('CUDA or MPS is not available. Training on CPU could be very slow.')
    
    model = segmentation_train(train_datasets = train_datasets, val_datasets = val_datasets, model = model, backbone = backbone, checkpoint = checkpoint, weights = weights, model_file = model_file, epochs = epochs, batch_size = batch_size, augment = augment, repeat = repeat, classification = classification, num_classes = num_classes, y_nodata = y_nodata, less_metrics = less_metrics, lr = lr, num_workers = num_workers, **kwargs)
    return model
    
def test(test_datasets, model, batch_size = 32, num_workers = 0):
    """
    Tests segmentation model.
    
    Parameters
    ----------
    test_datasets : list or list of lists
        Test data generated by generate_tiles() function. Each dataset is a list of 3 elements: training data (x): file path or xarray.DataArray, target variable (y): file path or xarray.DataArray, split_names: string or list of strings or 'all' if you need to use the whole dataset. You can provide a list of datasets to test model on multiple datasets.
    model : torch.nn model or SklearnModel or path to a model file
        Model to test. You can pass the model object returned by `train()` function or file (*.ckpt or *.joblib) where model is stored.
    batch_size : int (default = 32)
        Number of samples used in one iteration.
    num_workers: int or 'auto' (default = 0)
        Number of parallel workers that will load the data. Set 'auto' to let RSP choose the optimal number of workers, set 0 to disable multiprocessing. Can increase training speed, but can also cause errors (e.g. pickling errors).
            
    Examples
    --------
        >>> x_out, y_out = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1], split_names = ['train', 'val', 'test'])
        >>> train_ds = [x_out, y_out[0], 'train']
        >>> val_ds = [x_out, y_out[0], 'val']
        >>> test_ds = [x_out, y_out[0], 'test']
        >>> model = rsp.segmentation.train(train_ds, val_ds, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, batch_size = 32)
        ...
        >>> rsp.segmentation.test(test_ds, model = model, batch_size = 32)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃        Test metric        ┃       DataLoader 0        ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │      test_acc_epoch       │    0.8231202960014343     │
        │     test_auroc_epoch      │    0.7588028311729431     │
        │      test_iou_epoch       │    0.69323649406433105    │
        │      test_loss_epoch      │    0.40799811482429504    │
        │   test_precision_epoch    │    0.8231202960014343     │
        │     test_recall_epoch     │    0.8231202960014343     │
        └───────────────────────────┴───────────────────────────┘
    """
    # Type checking
    if not isinstance(test_datasets[0], list):
        test_datasets = [test_datasets]
    for i in test_datasets:
        if len(i) != 3:
            raise ValueError("Every dataset must consist of x, y and names")
        if not isinstance(i[0], str) and not isinstance(i[0], xarray.DataArray):
            raise TypeError("x in dataset must be a string or xarray.DataArray")
        elif isinstance(i[0], str) and not os.path.exists(i[0]):
            raise OSError(i + " does not exist")
        if not isinstance(i[1], str) and not isinstance(i[1], xarray.DataArray):
            raise TypeError("y in dataset must be a string or xarray.DataArray")
        elif isinstance(i[1], str) and not os.path.exists(i[1]):
            raise OSError(i + " does not exist")
        if not isinstance(i[2], str) and not isinstance(i[2], list):
            raise TypeError("name in dataset must be a string or a list")
    if not isinstance(batch_size, int):
        if isinstance(batch_size, type(None)):
            batch_size = 32
        else:
            raise TypeError("batch_size must be an integer")
    if (not isinstance(num_workers, int) and num_workers != 'auto') or (isinstance(num_workers, int) and num_workers < 0):
        if isinstance(num_workers, type(None)):
            num_workers = 'auto'
        else:
            raise TypeError("num_workers must be non-negative integer or 'auto'")
    
    cuda = cuda_test()
    if cuda == False:
        warnings.warn('CUDA or MPS is not available. Testing on CPU could be very slow.')
    
    segmentation_test(test_datasets = test_datasets, model = model, batch_size = batch_size, num_workers = num_workers)
    
 
def generate_map(x, y, reference, model, output, batch_size = 32, num_workers = 0, nodata = None,):
    """
    Create map using pre-trained model.
    
    Parameters
    ----------
    x : path as a string or xarray.DataArray
        Training data (x) generated by generate_tiles() function that will be used for prediction.
    y : path as a string or xarray.DataArray
        Target variable data (y) generated by generate_tiles() function that was used to train the model.
    reference : path as a string
        Raster that will be used as a reference raster to get size, transform and crs for a map. Use one of the rasters that were used for tile generation.
    model : torch.nn model or SklearnModel or path to a model file
        Pre-trained model to predict target values.  You can pass the model object returned by `train()` function or file (*.ckpt or *.joblib) where model is stored.
    output : path as a string
        Path where to write output map.
    batch_size : int (default = 32)
        Number of samples used in one iteration.
    num_workers: int or 'auto' (default = 0)
        Number of parallel workers that will load the data. Set 'auto' to let RSP choose the optimal number of workers, set 0 to disable multiprocessing. Can increase training speed, but can also cause errors (e.g. pickling errors).
    nodata : int or float (optional)
        Nodata value. If not defined then nodata value of y dataset will be used.
    
    Examples
    --------
        >>> x_out, y_out = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1], split_names = ['train', 'val', 'test'])
        >>> train_ds = [x_out, y_out[0], 'train']
        >>> val_ds = [x_out, y_out[0], 'val']
        >>> model = rsp.segmentation.train(train_ds, val_ds, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, batch_size = 32)
        ...
        >>> reference = '/home/rsp_test/mosaics/landcover.tif'
        >>> output_map = '/home/rsp_test/prediction.tif'
        >>> rsp.segmentation.generate_map(x_out, y_out[0], reference, model, output_map)
        Predicting: 100% #################### 372/372 [32:16, 1.6s/it]
        
        >>> x_file = '/home/rsp_test/model/x.zarr'
        >>> y_file = '/home/rsp_test/model/y.zarr'
        >>> model = '/home/rsp_test/model/upernet.ckpt'
        >>> reference = '/home/rsp_test/mosaics/landcover.tif'
        >>> output_map = '/home/rsp_test/prediction.tif'
        >>> rsp.segmentation.generate_map(x_file, y_file, reference, model, output_map)
        Predicting: 100% #################### 372/372 [32:16, 1.6s/it]
        
        >>> # Train model on data from Montana
        >>> x_montana_files = glob('/home/rsp_test/mosaics/landsat_montana/*')
        >>> y_montana_files = '/home/rsp_test/mosaics/landcover_montana/landcover.tif'
        >>> x_montana, y_montana = rsp.segmentation.generate_tiles(x_montana_files, y_montana_files, tile_size = 256, shuffle = True, split = [3, 1, 1], split_names = ['train', 'val', 'test'])
        >>> train_ds = [x_montana, y_montana[0], 'train']
        >>> val_ds = [x_montana, y_montana[0], 'val']
        >>> model_montana = rsp.segmentation.train(train_ds, val_ds, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, batch_size = 32)
        ...
        >>> # Use model to map landcover of Idaho
        >>> x_idaho_files = glob('/home/rsp_test/mosaics/landsat_idaho/*')
        >>> x_idaho, _ = rsp.segmentation.generate_tiles(x_idaho_files, None, tile_size = 256)
        >>> reference = x_idaho_files[0]
        >>> output_map = '/home/rsp_test/prediction_idaho.tif'
        >>> rsp.segmentation.generate_map(x_idaho, y_montana, reference, model_montana, output_map)
        Predicting: 100% #################### 372/372 [32:16, 1.6s/it]
    """
    # Type checking
    if not isinstance(x, str) and not isinstance(x, xarray.DataArray):
        raise TypeError("x must be a string or xarray.DataArray")
    elif isinstance(x, str) and not os.path.exists(x):
        raise OSError(x + " does not exist")
    if not isinstance(y, str) and not isinstance(y, xarray.DataArray):
        raise TypeError("y must be a string or xarray.DataArray")
    elif isinstance(y, str) and not os.path.exists(y):
        raise OSError(x + " does not exist")
    if not isinstance(reference, str):
        raise TypeError("reference must be a string")
    elif not os.path.exists(reference):
        raise OSError(reference + " does not exist")
    if not isinstance(output, str):
        raise TypeError("output must be a string")
    if not isinstance(batch_size, int):
        if isinstance(batch_size, type(None)):
            batch_size = 32
        else:
            raise TypeError("batch_size must be an integer")
    if (not isinstance(num_workers, int) and num_workers != 'auto') or (isinstance(num_workers, int) and num_workers < 0):
        if isinstance(num_workers, type(None)):
            num_workers = 'auto'
        else:
            raise TypeError("num_workers must be non-negative integer or 'auto'")
    if not isinstance(nodata, int) and not isinstance(nodata, float) and not isinstance(nodata, type(None)):
        raise TypeError("nodata must be integer or float")
    
    cuda = cuda_test()
    if cuda == False and superres == True:
        warnings.warn('CUDA or MPS is not available. Prediction on CPU could be very slow.')
    
    predict_map_from_tiles(x = x, y = y, reference = reference, model = model, output = output, nodata = nodata, batch_size = batch_size, num_workers = num_workers)

