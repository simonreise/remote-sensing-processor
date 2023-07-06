import sys

from remote_sensing_processor.segmentation.segmentation import segmentation_train, segmentation_test
from remote_sensing_processor.segmentation.tiles import get_ss_tiles
from remote_sensing_processor.segmentation.mapping import predict_map_from_tiles


def generate_tiles(x, y, tile_size = 128, classification = True, shuffle = False, samples_file = None, split = [1], x_outputs = None, y_outputs = None, x_dtype = None, y_dtype = None, x_nodata = None, y_nodata = None):
    """
    Cut rasters into tiles
    
    Parameters
    ----------
    x : list of paths as strings
        Rasters to use as training data.
    y : path as a string
        Raster to use as target values.
    tile_size : int (default = 128)
        Size of tiles to generate (tile_size x tile_size).
    classification : bool (default = True)
        If True then tiles will be prepared for classification (e.g. semantic segmentation) task, else will be prepared for regression task.
    shuffle : bool (default = False)
        Is random shuffling of samples needed.
    samples_file : path as a string (optional)
        Path where to save tiles, samples and classes data that are generated as output. File should have .pickle format. It can be needed later for mapping.
    split : list of ints (optional)
        Splitting data in subsets. Is a list of integers defining proportions of every subset. [3, 1, 1] will generate 3 subsets in proportion 3 to 1 to 1.
    x_outputs : list of paths as strings (optional)
        List of paths to save generated output x data. Data is saved in .h5 format.
    y_outputs : list of paths as strings (optional)
        List of paths to save generated output y data. Data is saved in .h5 format.
    x_dtype : dtype definition as a string (optional)
        If you run out of memory, you can try to convert your data to less memory consuming format.
    y_dtype : dtype definition as a string (optional)
        If you run out of memory, you can try to convert your data to less memory consuming format.
    x_nodata : int or float (optional)
        You can define which value in x raster corresponds to nodata and areas that contain nodata in x raster will be ignored while training and testing. Tiles that contain only nodata in both x and y will be omited.
    y_nodata : int or float (optional)
        You can define which value in y raster corresponds to nodata and areas that contain nodata in y raster will be ignored while training and testing. Tiles that contain only nodata in both x and y will be omited.
    
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
        classification : bool
            If True then tiles are prepared for classification (e.g. semantic segmentation) task, else are prepared for regression task.
        num_classes : int
            Number of classes for classification task.
        classes : list
            Sorted unique values from y dataset.
        x_nodata : int or float 
            Nodata value for x tiles. If was not set as patameter then is obtained from x metadata.
        y_nodata : int or float 
            Nodata value for y tiles or nodata class index if y if task is classification. If was not set as patameter then is obtained from y metadata.
            
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
        >>> x_train_file = '/home/rsp_test/model/x_train.h5'
        >>> x_val_file = '/home/rsp_test/model/x_val.h5'
        >>> x_test_file = '/home/rsp_test/model/x_test.h5'
        >>> y_train_file = '/home/rsp_test/model/y_train.h5'
        >>> y_val_file = '/home/rsp_test/model/y_val.h5'
        >>> y_test_file = '/home/rsp_test/model/y_test.h5'
        >>> x_i, y_i, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, samples_file = s_file, split = [3, 1, 1], x_outputs = [x_train_file, x_val_file, x_test_file], y_outputs = [y_train_file, y_val_file, y_test_file], x_nodata = 0, y_nodata = 0)
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
        >>> print(classification)
        True
        >>> print(num_classes)
        11
        >>> print(classes)
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        >>> print(x_nodata)
        0
        >>> print(y_nodata)
        0
    """
    x, y, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata = get_ss_tiles(x = x, y = y, tile_size = tile_size, classification = classification, shuffle = shuffle, samples_file = samples_file, split = split, x_outputs = x_outputs, y_outputs = y_outputs, x_dtype = x_dtype, y_dtype = y_dtype, x_nodata = x_nodata, y_nodata = y_nodata)
    if len(split) == 1:
        x = x[0]
        y = y[0]
    return x, y, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata

    
def train(x_train, y_train, x_val, y_val, model_file, model, backbone = None, epochs = 5, checkpoint = None, weights = None, batch_size = 32, less_metrics = False, lr = 1e-3, classification = None, num_classes = None, x_nodata = None, y_nodata = None):
    """
    Trains segmentation model
    
    Parameters
    ----------
    x_train : path as a string or numpy array
        Training tiles generated by generate_tiles() function.
    y_train : path as a string or numpy array
        Training tiles generated by generate_tiles() function.
    x_val : path as a string or numpy array
        Validation tiles generated by generate_tiles() function. Can be set to None if no validation needed.
    y_val : path as a string or numpy array
        Validation tiles generated by generate_tiles() function. Can be set to None if no validation needed.
    model_file : path as a string
        Checkpoint file (*.ckpt) where model will be saved after training.
    model : str
        Name of model architecture.
    backbone : str (optional)
        Backbone of a model, if multiple backbones are supported.
    checkpoint : path as a string (optional)
        Checkpoint file (*.ckpt) of a pre-trained model to fine-tune.
    weights : str
        Name of pre-trained weights to fine-tune.
    epochs : int (default = 5)
        Number of training epochs.
    batch_size : int (default = 32)
        Number of training samples used in one iteration.
    less_metrics : bool (default = False)
        Sometimes Torchmetrics can freeze while calculating precision, recall and IOU. If it happens, try restarting with `less_metrics = True`.
    lr : float (default = 1e-3)
        Learning rate of a model. Lower value results usually in better model convergence, but much slower training.
    classification : bool (default = None)
        If True then tiles are prepared for classification (e.g. semantic segmentation) task, else are prepared for regression task. If not stated then is read from y_train h5 file or set to True if y_train is np.array.
    num_classes: int (optional)
        Number of classes for classification task. If not stated then is read from y_train h5 file or is set to np.max + 1.
    x_nodata : int or float (optional)
        You can define which value in x raster corresponds to nodata and areas that contain nodata in x raster will be ignored while training and testing. If not stated then is read from x_train h5 file.
    y_nodata : int or float (optional)
        You can define which value in y raster corresponds to nodata and areas that contain nodata in y raster will be ignored while training and testing. If not stated then is read from y_train h5 file.
    
    Returns
    ----------
    torch.nn model
        Trained model.
            
    Examples
    --------
        >>> x_i, y_i, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1])
        >>> x_train = x_i[0]
        >>> x_val = x_i[1]
        >>> x_test = x_i[2]
        >>> y_train = x_i[0]
        >>> y_val = x_i[1]
        >>> y_test = x_i[2]
        >>> model = rsp.segmentation.train(x_train, y_train, x_val, y_val, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, batch_size = 32, classification = classification, num_classes = num_classes, x_nodata = x_nodata, y_nodata = y_nodata)
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
        
        >>> x_train = '/home/rsp_test/model/x_train.h5'
        >>> x_val = '/home/rsp_test/model/x_val.h5'
        >>> y_train = '/home/rsp_test/model/y_train.h5'
        >>> y_val = '/home/rsp_test/model/y_val.h5'
        >>> model = rsp.segmentation.train(x_train, y_train, x_val, y_val, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, batch_size = 32)
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
    """
    model = segmentation_train(x_train = x_train, x_val = x_val, y_train = y_train, y_val = y_val, model = model, backbone = backbone, checkpoint = checkpoint, weights = weights, model_file = model_file, epochs = epochs, batch_size = batch_size, classification = classification, num_classes = num_classes, x_nodata = x_nodata, y_nodata = y_nodata, less_metrics = less_metrics, lr = lr)
    return model
    
def test(x_test, y_test, model, batch_size = 32):
    """
    Tests segmentation model
    
    Parameters
    ----------
    x_test : path as a string or numpy array
        Training tiles generated by generate_tiles() function.
    y_test : path as a string or numpy array
        Training tiles generated by generate_tiles() function.
    model : torch.nn model or model name
        Model to test. You can pass the model object returned by `train()` function or file (*.ckpt) where model is stored.
    batch_size : int (default = 32)
        Number of samples used in one iteration.
            
    Examples
    --------
        >>> x_i, y_i, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1])
        >>> x_train = x_i[0]
        >>> x_val = x_i[1]
        >>> x_test = x_i[2]
        >>> y_train = x_i[0]
        >>> y_val = x_i[1]
        >>> y_test = x_i[2]
        >>> model = rsp.segmentation.train(x_train, y_train, x_val, y_val, model = 'UperNet', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, batch_size = 32, classification = classification, num_classes = num_classes, x_nodata = x_nodata, y_nodata = y_nodata)
        ...
        >>> rsp.segmentation.test(x_test, y_test, model = model, batch_size = 32)
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
    segmentation_test(x_test = x_test, y_test = y_test, model = model, batch_size = batch_size)
    
 
def generate_map(x, y_true, model, output, tiles = None, samples = None, classes = None, samples_file = None, nodata = None, batch_size = 32):
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
        List of tile coordinates. Usually is generated in `generate_tiles` function. You also can use `samples_file` instead.
    samples : list (optional) 
        List with order of samples. Usually is generated in `generate_tiles` function. You also can use `samples_file` instead.
    classes : list
        Sorted unique values from y dataset. Usually is generated in `generate_tiles` function. You also can use `samples_file` instead.
    samples_file : path as a string (optional)
        Path to a samples .pickle file generated by `generate_tiles` function. You can use `samples_file` instead of `tiles`, `samples` and `classes`.
    nodata : int or float (optional)
        Nodata value.
    batch_size : int (default = 32)
        Number of samples used in one iteration.
    
    Examples
    --------
        >>> x_i, y_i, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1])
        >>> x_train = x_i[0]
        >>> x_val = x_i[1]
        >>> x_test = x_i[2]
        >>> y_train = x_i[0]
        >>> y_val = x_i[1]
        >>> y_test = x_i[2]
        >>> model = rsp.segmentation.train(x_train, y_train, x_val, y_val, model = 'UperNet', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, batch_size = 32, classification = classification, num_classes = num_classes, x_nodata = x_nodata, y_nodata = y_nodata)
        ...
        >>> y_reference = '/home/rsp_test/mosaics/landcover.tif'
        >>> output_map = '/home/rsp_test/prediction.tif'
        >>> rsp.segmentation.generate_map([x_train, x_val, x_test], y_reference, model, output_map, tiles = tiles, samples = samples, classes = classes, nodata = y_nodata)
        Predicting: 100% #################### 372/372 [32:16, 1.6s/it]
        
        >>> x_train_file = '/home/rsp_test/model/x_train.h5'
        >>> x_val_file = '/home/rsp_test/model/x_val.h5'
        >>> x_test_file = '/home/rsp_test/model/x_test.h5'
        >>> s_file = '/home/rsp_test/model/samples.pickle'
        >>> model = '/home/rsp_test/model/upernet.ckpt'
        >>> y_reference = '/home/rsp_test/mosaics/landcover.tif'
        >>> output_map = '/home/rsp_test/prediction.tif'
        >>> rsp.segmentation.generate_map([x_train_file, x_val_file, x_test_file], y_reference, model, output_map, samples_file = s_file, nodata = -1)
        Predicting: 100% #################### 372/372 [32:16, 1.6s/it]
    """
    if (tiles != None and samples != None) or (samples_file != None):
        if not isinstance(x, list):
            x = [x]
        predict_map_from_tiles(x = x, y_true = y_true, model = model, tiles = tiles, samples = samples, classes = classes, samples_file = samples_file, output = output, nodata = nodata, batch_size = batch_size)
    else:
        print('Tiles and samples must be specified')

