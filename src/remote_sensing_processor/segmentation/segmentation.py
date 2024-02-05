import os
import collections
import joblib
import warnings

import numpy as np
import xarray
import dask

import sklearn.metrics
import torch
import torchvision
from torchvision.transforms import v2
from torchvision import tv_tensors
import torchmetrics
import transformers
import lightning as l

from remote_sensing_processor.common.common_functions import PersistManager

from remote_sensing_processor.segmentation.models import load_model, load_sklearn_model


# These warnings usually appear on sanity check if a loaded tile is empty
warnings.filterwarnings("ignore", message = "No positive samples in targets")
warnings.filterwarnings("ignore", message = "exists and is not empty")
warnings.filterwarnings("ignore", message = "could not find the monitored key in the returned metrics")
warnings.filterwarnings("ignore", message = "Skipping val loop")

dask.config.set(scheduler='synchronous')


def segmentation_train(train_datasets, val_datasets, model, backbone, checkpoint, weights, model_file, epochs, batch_size, augment, repeat, classification, num_classes, y_nodata, less_metrics, lr, num_workers):
    input_shape = None
    input_dims = None
    # Deep learning pytorch models
    if model in ['BEiT', 'ConditionalDETR', 'Data2Vec', 'DETR', 'DPT', 'Mask2Former', 'MaskFormer', 'MobileNetV2', 'MobileViT', 'MobileViTV2', 'OneFormer', 'SegFormer', 'UperNet', 'DeepLabV3', 'FCN', 'LRASPP']:
        # Checking if file extention is right
        if os.path.splitext(model_file)[1] != '.ckpt':
            raise ValueError("Wrong model file format: .ckpt file extention expected for " + model)
        # Setting up data module
        dm = SegDataModule(train_datasets = train_datasets, val_datasets = val_datasets, repeat = repeat, augment = augment, batch_size = batch_size, num_workers = num_workers)
        # Reading parameters that are needed to build the model
        if input_shape == None:
            input_shape = dm.input_shape
        if input_dims == None:
            input_dims = dm.input_dims
        if num_classes == None:
            num_classes = dm.num_classes
        if classification == None:
            classification = dm.classification
        if y_nodata == None:
            y_nodata = dm.y_nodata
        # Loading model
        if checkpoint != None:
            model = Model.load_from_checkpoint(checkpoint, input_shape = input_shape, input_dims = input_dims, num_classes = num_classes, classification = classification, y_nodata = y_nodata, lr = lr)
        else:
            model = Model(model, backbone, weights, input_shape, input_dims, num_classes, classification, y_nodata, less_metrics, lr)
        # Setting up trainer
        checkpoint_callback = l.pytorch.callbacks.ModelCheckpoint(
            save_top_k = 1,
            monitor = "val_loss" if not isinstance(val_datasets, type(None)) else "train_loss",
            mode = "min",
            dirpath = os.path.dirname(model_file),
            filename = os.path.basename(os.path.splitext(model_file)[0]),
            enable_version_counter = False
        )
        tb_logger = l.pytorch.loggers.TensorBoardLogger(save_dir = os.path.join(os.path.dirname(model_file), 'logs' , 'tensorboard'))
        csv_logger = l.pytorch.loggers.CSVLogger(save_dir = os.path.join(os.path.dirname(model_file), 'logs' , 'csv'))
        trainer = l.Trainer(max_epochs = epochs, callbacks = [checkpoint_callback], logger = [tb_logger, csv_logger])
        # Training
        trainer.fit(model, dm)
    # Sklearn ML models
    elif model in ["Nearest Neighbors", "Logistic Regression", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Gradient Boosting", "Multilayer Perceptron", "AdaBoost", "Naive Bayes", "QDA", "Ridge", "Lasso", "ElasticNet", "XGBoost", "XGB Random Forest"]:
        # Checking if file extention is right
        if os.path.splitext(model_file)[1] != '.joblib':
            raise ValueError("Wrong model file format: .joblib file extention expected for " + model)
        # Setting up persist manager
        pm = PersistManager()
        # Loading train datasets
        x_train, y_train, classification, y_nodata, num_classes, pm = sklearn_load_dataset(train_datasets, pm)
        if checkpoint != None:
            model = joblib.load(checkpoint)
            if model.model_name in ["Random Forest", "Gradient Boosting"]:
                model.model.n_estimators += 50
        else:
            model = SklearnModel(model, backbone, classification, epochs, y_nodata, num_classes)
        model.fit(x_train, y_train)
        del x_train
        del y_train
        # Validation
        if not isinstance(val_datasets, type(None)):
            x_val, y_val, _, _, _, pm = sklearn_load_dataset(val_datasets, pm)
            model.test(x_val, y_val)
        try:
            joblib.dump(model, model_file, compress = 9)
        except:
            warnings.warn('Error while saving model, check if enough free space is available.')
    else:
        raise ValueError("Wrong model name. Check spelling or read a documentation and choose a supported model")
    return model

    
def segmentation_test(test_datasets, model, batch_size, num_workers):
    for ds in test_datasets:
        if len(ds) != 3:
            raise ValueError("Every dataset must consist of x, y and names")  
    # Loading model
    if isinstance(model, str):
        if '.ckpt' in model:
            model = Model.load_from_checkpoint(model)
        elif '.joblib' in model:
            model = joblib.load(model)
    # Neural networks
    if model.model_name in ['BEiT', 'ConditionalDETR', 'Data2Vec', 'DETR', 'DPT', 'Mask2Former', 'MaskFormer', 'MobileNetV2', 'MobileViT', 'MobileViTV2', 'OneFormer', 'SegFormer', 'UperNet', 'DeepLabV3', 'FCN', 'LRASPP']:
        dm = SegDataModule(test_datasets = test_datasets, batch_size = batch_size, num_workers = num_workers)
        trainer = l.Trainer()
        trainer.test(model, dm)
    # Sklearn models
    elif model.model_name in ["Nearest Neighbors", "Logistic Regression", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Gradient Boosting", "Multilayer Perceptron", "AdaBoost", "Naive Bayes", "QDA", "Ridge", "Lasso", "ElasticNet", "XGBoost", "XGB Random Forest"]:
        classification = model.classification
        pm = PersistManager()
        # Loading test datasets
        x_test, y_test, _, _, _, pm = sklearn_load_dataset(test_datasets, pm)
        model.test(x_test, y_test)
    else:
        raise ValueError("Wrong model name. Check spelling or read a documentation and choose a supported model")


class ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, names, transform):
        # Reading x dataset
        if isinstance(x, str):
            self.x_dataset = xarray.open_dataarray(x, engine = 'zarr', chunks = 'auto', mask_and_scale = False)
        elif isinstance(x, xarray.DataArray):
            self.x_dataset = x
        # Getting tiles and samples
        self.tiles = self.x_dataset.tiles
        self.border = self.x_dataset.border
        samples = []
        for i in range(len(self.x_dataset.names)):
            if names == 'all' or self.x_dataset.names[i] in names:
                samples.extend(self.x_dataset.samples[i])
        self.samples = samples
        # Getting indices of samples
        indices = []
        for k, v in enumerate([x for xs in self.x_dataset.samples for x in xs]):
            if v in samples:
                indices.append(k)
        self.indices = indices
        # Getting len
        self.dataset_len = len(self.samples)
        if not isinstance(y, type(None)):
            # Reading y dataset
            if isinstance(y, str):
                self.y_dataset = xarray.open_dataarray(y, engine = 'zarr', chunks = 'auto', mask_and_scale = False)
            elif isinstance(y, xarray.DataArray):
                self.y_dataset = y
        else:
            self.y_dataset = None
        # Setting up transform
        self.transform = transform
        self.pm = PersistManager()
        self.pm.is_persisted = False

    def __getitem__(self, index):
        # Persist is in getitem because if it is in init then multiprocessing is not working because datasets are too big to pickle
        if self.pm.is_persisted == False:
            self.x_dataset = self.pm.persist(self.x_dataset)
            if not isinstance(self.y_dataset, type(None)):
                self.y_dataset = self.pm.persist(self.y_dataset)
            self.pm.is_persisted = True
        x = tv_tensors.Image(self.x_dataset[self.indices[index]].data.astype('float32').compute())
        if not isinstance(self.y_dataset, type(None)):
            y = tv_tensors.Mask(self.y_dataset[self.indices[index]].data.compute())
            # Transform
            if self.transform != None:
                x, y = self.transform(x, y)
            return x, y
        else:
            # Transform
            if self.transform != None:
                x = self.transform(x)
            return x

    def __len__(self):
        return self.dataset_len

    def __del__(self):
        self.x_dataset.close()
        if not isinstance(self.y_dataset, type(None)):
            self.y_dataset.close()
            

class SegDataModule(l.LightningDataModule):
    def __init__(self, train_datasets = None, val_datasets = None, test_datasets = None, pred_dataset = None, repeat = 1, augment = False, batch_size = 32, num_workers = 'auto'):
        super().__init__()
        # Setting up datasets
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        self.pred_dataset = pred_dataset
        self.repeat = repeat
        self.batch_size = batch_size
        # Configuring multiporcessing
        if num_workers != 0 and num_workers != 'auto' and num_workers > torch.multiprocessing.cpu_count():
            warnings.warn("'num_workers' is " + num_workers + ", but you have only " + torch.multiprocessing.cpu_count() + "CPU cores. Setting 'num_workers' to 'auto'")
            num_workers = 'auto'
        if num_workers == 'auto':
            cpus = torch.multiprocessing.cpu_count()
            gpus = max(torch.cuda.device_count(), 1)
            self.workers = max(1 , cpus // gpus - 1)
            self.pw = True
        elif num_workers != 0:
            self.workers = num_workers
            self.pw = True
        else:
            self.workers = 0
            self.pw = False
        # Parameters that are needed to build the model
        self.input_shape = None
        self.input_dims = None
        self.num_classes = None
        self.classification = None
        self.x_nodata = None
        self.y_nodata = None
        self.classes = None
        # Dataset check
        if self.train_datasets:
            self.dataset_check(self.train_datasets)
        if self.val_datasets:
            self.dataset_check(self.val_datasets)
        if self.test_datasets:
            self.dataset_check(self.test_datasets)
        if self.pred_dataset:
            self.dataset_check([[self.pred_dataset, None, 'all']])
        # Setting up transform
        if augment == True:
            self.transform = v2.Compose([
                v2.RandomResizedCrop(size = (self.input_shape, self.input_shape), antialias = True),
                v2.RandomHorizontalFlip(p = 0.5),
                # Removed because of nans
                #v2.RandomRotation(45, fill = {tv_tensors.Image: self.x_nodata.item(), tv_tensors.Mask: self.y_nodata.item()}),
                #v2.ElasticTransform(fill = {tv_tensors.Image: self.x_nodata.item(), tv_tensors.Mask: self.y_nodata.item()}),
                ])
        else:
            self.transform = None
    
    def dataset_check(self, datasets):
        for ds in datasets:
            if isinstance(ds[0], str):
                x_dataset = xarray.open_dataarray(ds[0], engine = 'zarr', mask_and_scale = False)
            elif isinstance(ds[0], xarray.DataArray):
                x_dataset = ds[0]
            input_shape = x_dataset.shape[2]
            input_dims = x_dataset.shape[1]
            x_nodata = x_dataset.rio.nodata
            if not isinstance(ds[1], type(None)):
                if isinstance(ds[1], str):
                    y_dataset = xarray.open_dataarray(ds[1], engine = 'zarr', mask_and_scale = False)
                elif isinstance(ds[1], xarray.DataArray):
                    y_dataset = ds[1]
                y_nodata = y_dataset.rio.nodata
                classification = y_dataset.classification
                classes = y_dataset.classes
                num_classes = y_dataset.num_classes
                assert y_dataset.shape[1] == input_shape
                assert x_dataset.tiles == y_dataset.tiles
                assert x_dataset.samples == y_dataset.samples
                assert x_dataset.border == y_dataset.border
            else:
                y_nodata = None
                classification = None
                classes = None
                num_classes = None
            if self.input_shape is not None:
                assert input_shape == self.input_shape
            elif input_shape is not None:
                self.input_shape = input_shape
            if self.input_dims is not None:
                assert input_dims == self.input_dims
            elif input_dims is not None:
                self.input_dims = input_dims
            if self.x_nodata is not None:
                assert x_nodata == self.x_nodata
            elif x_nodata is not None:
                self.x_nodata = x_nodata
            if self.y_nodata is not None:
                assert y_nodata == self.y_nodata
            elif y_nodata is not None:
                self.y_nodata = y_nodata
            if self.classification is not None:
                assert classification == self.classification
            elif classification is not None:
                self.classification = classification
            if self.classes is not None:
                assert classes == self.classes
            elif classes is not None:
                self.classes = classes
            if self.num_classes is not None:
                assert num_classes == self.num_classes
            elif num_classes is not None:
                self.num_classes = num_classes

    def setup(self, stage):
        if stage == 'fit':
            datasets = []
            for ds in self.train_datasets:
                for j in range(self.repeat):
                    datasets.append(ZarrDataset(ds[0], ds[1], ds[2], transform = self.transform))
            self.ds_train = torch.utils.data.ConcatDataset(datasets)
            if not isinstance(self.val_datasets, type(None)):
                datasets = []
                for ds in self.val_datasets:
                    datasets.append(ZarrDataset(ds[0], ds[1], ds[2], transform = None))
                self.ds_val = torch.utils.data.ConcatDataset(datasets)
            else:
                self.ds_val = None
        if stage == 'test':
            datasets = []
            for ds in self.test_datasets:
                datasets.append(ZarrDataset(ds[0], ds[1], ds[2], transform = None))
            self.ds_test = torch.utils.data.ConcatDataset(datasets)
        if stage == 'predict':
            self.ds_pred = ZarrDataset(self.pred_dataset, None, 'all', transform = None)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_train, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = self.workers, persistent_workers = self.pw)
        #return DataLoader2(self.ds_train, reading_service = self.rs)

    def val_dataloader(self):
        if not isinstance(self.ds_val, type(None)):
            return torch.utils.data.DataLoader(self.ds_val, batch_size = self.batch_size, pin_memory = True, num_workers = self.workers, persistent_workers = self.pw)
            #return DataLoader2(self.ds_val, reading_service = self.rs)
        else:
            return None

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_test, batch_size = self.batch_size, pin_memory = True, num_workers = self.workers, persistent_workers = self.pw)
        #return DataLoader2(self.ds_test, reading_service = self.rs)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_pred, batch_size = self.batch_size, pin_memory = True, num_workers = self.workers, persistent_workers = self.pw)
        #return DataLoader2(self.ds_pred, reading_service = self.rs)


class Model(l.LightningModule):
    def __init__(
        self,
        model,
        backbone,
        weights,
        input_shape,
        input_dims,
        num_classes,
        classification,
        y_nodata,
        less_metrics,
        lr
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model
        self.model = load_model(model, backbone, weights, input_shape, input_dims, num_classes)
        self.classification = classification
        self.less_metrics = less_metrics
        self.num_classes = num_classes
        self.y_nodata = y_nodata
        if classification:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index = int(y_nodata))
        self.lr = lr
        torch.autograd.set_detect_anomaly(True)
    
    def forward(self, x):
        if isinstance(self.model, transformers.OneFormerForUniversalSegmentation):
            # Oneformer also requires tokenized tasks as inputs, task is semantic
            t = torch.tensor([[49406,   518, 10549,   533, 29119,  1550, 49407,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0]], device = self.device)
            t = torch.cat([t] * x.shape[0], dim=0)
            pred = self.model(x, t)
        else: 
            pred = self.model(x)
        return pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        y = self.ytype(y)
        pred = self.process_output(pred, x)
        #print(torch.unique(torch.isnan(pred)))
        #print(y.min(), y.max())
        #print(pred.min(), pred.max())
        if self.classification:
            loss = self.loss_fn(pred, y)
        else:
            loss = self.mse_loss(pred.squeeze(), y.squeeze(), ignore_index = self.y_nodata)
        self.log_all(y, pred, loss, 'train')
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        y = self.ytype(y)
        pred = self.process_output(pred, x)
        if self.classification:
            loss = self.loss_fn(pred, y)
        else:
            loss = self.mse_loss(pred.squeeze(), y.squeeze(), ignore_index = self.y_nodata)
        self.log_all(y, pred, loss, 'val')
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        y = self.ytype(y)
        pred = self.process_output(pred, x)
        if self.classification:
            loss = self.loss_fn(pred, y)
        else:
            loss = self.mse_loss(pred.squeeze(), y.squeeze(), ignore_index = self.y_nodata)
        self.log_all(y, pred, loss, 'test')
    
    def log_all(self, y, pred, loss, stage):
        self.log(stage + '_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        if self.classification:
            # Acc average is micro because most of papers use micro and sklearn uses micro
            self.log(stage + '_acc', torchmetrics.functional.classification.multiclass_accuracy(pred, y, num_classes = self.num_classes, average = 'micro', ignore_index = int(self.y_nodata)), on_step = True, on_epoch = True, prog_bar = True)
            self.log(stage + '_auroc', torchmetrics.functional.classification.multiclass_auroc(pred, y, num_classes = self.num_classes, average = 'macro', ignore_index = int(self.y_nodata)), on_step = True, on_epoch = True, prog_bar = True)
            # TODO: these metrics somehow make detr freeze after first steps, hope this is temporary fix
            if not self.less_metrics:
                self.log(stage + '_precision', torchmetrics.functional.classification.multiclass_precision(pred, y, num_classes = self.num_classes, average = 'macro', ignore_index = int(self.y_nodata)), on_step = True, on_epoch = True)
                self.log(stage + '_recall', torchmetrics.functional.classification.multiclass_recall(pred, y, num_classes = self.num_classes, average = 'macro', ignore_index = int(self.y_nodata)), on_step = True, on_epoch = True)
                self.log(stage + '_iou', torchmetrics.functional.classification.multiclass_jaccard_index(pred, y, num_classes = self.num_classes, average = 'macro', ignore_index = int(self.y_nodata)), on_step = True, on_epoch = True, prog_bar = True)
        else:
            filtered = torch.where(torch.reshape(y, (y.shape[0], -1)) != self.y_nodata, True, False).nonzero().flatten().tolist()
            self.log(stage + '_r2', torchmetrics.functional.r2_score(torch.reshape(torch.squeeze(pred), (pred.shape[0], -1))[filtered], torch.reshape(y, (y.shape[0], -1))[filtered]), on_step = True, on_epoch = True, prog_bar = True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        x = batch
        pred = self.forward(x)
        pred = self.process_output(pred, x)
        if self.classification:
            pred = pred.argmax(dim = 1)
        else:
            pred = pred.squeeze()
        return pred
    
    def process_output(self, pred, x):
        if isinstance(pred, transformers.modeling_outputs.SemanticSegmenterOutput) or isinstance(pred, transformers.models.clipseg.modeling_clipseg.CLIPSegImageSegmentationOutput):
            pred = pred.logits
            if pred.shape[2:4] != x.shape[2:4]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size = x.shape[2:4],
                    mode = "bilinear",
                    align_corners = False,
                )
        elif isinstance(pred, transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrSegmentationOutput) or isinstance(pred, transformers.models.detr.modeling_detr.DetrSegmentationOutput):
            class_queries_logits = pred.logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = pred.pred_masks  # [batch_size, num_queries, height, width]
            
            # Remove the null class `[..., :-1]`
            # Detr treats 0 as separate class, conditionaldetr treats it as nullclass
            if isinstance(pred, transformers.models.detr.modeling_detr.DetrSegmentationOutput):
                masks_classes = class_queries_logits.softmax(dim = -1)[..., :-1]
            else:
                masks_classes = class_queries_logits.softmax(dim = -1)
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

            if pred.shape[2:4] != x.shape[2:4]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size = x.shape[2:4],
                    mode = "bilinear",
                    align_corners = False,
                )
        elif isinstance(pred, transformers.models.mask2former.modeling_mask2former.Mask2FormerForUniversalSegmentationOutput) or isinstance(pred, transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput):
            class_queries_logits = pred.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = pred.masks_queries_logits  # [batch_size, num_queries, height, width]
            masks_classes = class_queries_logits.softmax(dim = -1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]
            
            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            if pred.shape[2:4] != x.shape[2:4]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size = x.shape[2:4],
                    mode = "bilinear",
                    align_corners = False,
                )
        elif isinstance(pred, transformers.models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput):
            class_queries_logits = pred.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = pred.masks_queries_logits  # [batch_size, num_queries, height, width]
            masks_classes = class_queries_logits.softmax(dim = -1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            if pred.shape[2:4] != x.shape[2:4]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size = x.shape[2:4],
                    mode = "bilinear",
                    align_corners = False,
                )        
        elif isinstance(pred, collections.OrderedDict):
            pred = pred['out']
        return pred
    
    def ytype(self, y):
        if self.classification:
            y = y.long()
        else:
            y = y.float()
        return y
    
    def mse_loss(self, pred, target, ignore_index = 0.0, reduction = 'mean'):
        mask = target == ignore_index
        out = (pred[~mask] - target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr)
        return optimizer
        
        
class SklearnModel:
    def __init__(self, model, backbone, classification, epochs, y_nodata, num_classes):
        self.classification = classification
        self.model_name = model
        self.model = load_sklearn_model(model, backbone, classification, epochs)
        self.y_nodata = y_nodata
        self.num_classes = num_classes
    
    def fit(self, x, y):
        self.model.fit(x, y)
        self.test(x, y)
    
    def test(self, x, y):
        if self.classification:
            pred = self.model.predict_proba(x)
            # TODO: find a way to implement macro average accuracy
            print('Accuracy: ', torchmetrics.functional.classification.multiclass_accuracy(torch.Tensor(pred), torch.Tensor(y.data.compute()).long(), num_classes = self.num_classes, average = 'micro', ignore_index = int(self.y_nodata)).item())
            print('Precision: ', torchmetrics.functional.classification.multiclass_precision(torch.Tensor(pred), torch.Tensor(y.data.compute()).long(), num_classes = self.num_classes, average = 'macro', ignore_index = int(self.y_nodata)).item())
            print('Recall: ', torchmetrics.functional.classification.multiclass_recall(torch.Tensor(pred), torch.Tensor(y.data.compute()).long(), num_classes = self.num_classes, average = 'macro', ignore_index = int(self.y_nodata)).item())
            print('ROC_AUC: ', torchmetrics.functional.classification.multiclass_auroc(torch.Tensor(pred), torch.Tensor(y.data.compute()).long(), num_classes = self.num_classes, average = 'macro', ignore_index = int(self.y_nodata)).item())
            print('IOU: ', torchmetrics.functional.classification.multiclass_jaccard_index(torch.Tensor(pred), torch.Tensor(y.data.compute()).long(), num_classes = self.num_classes, average = 'macro', ignore_index = int(self.y_nodata)).item())
        else:
            filtered = np.where(y.data.compute() != self.y_nodata, True, False).nonzero()[0].tolist()
            pred = self.model.predict(x)
            print('R2: ', torchmetrics.functional.r2_score(torch.Tensor(pred[filtered]), torch.Tensor(y[filtered].data.compute())).item())
            print('MSE: ', torchmetrics.functional.mean_squared_error(torch.Tensor(pred[filtered]), torch.Tensor(y[filtered].data.compute())).item())
    
    def predict(self, x):
        pred = self.model.predict(x)
        return pred
        
    
def sklearn_load_dataset(ds, pm):
    x_stack = None
    y_stack = None
    classification = None
    y_nodata = None
    num_classes = None
    for d in ds:
        x = d[0]
        y = d[1]
        names = d[2]
        # Reading x dataset
        if isinstance(x, str):
            x_dataset = xarray.open_dataarray(x, engine = 'zarr', chunks = 'auto', mask_and_scale = False)
        else:
            x_dataset = x
        x_dataset = pm.persist(x_dataset)
        # Reading y dataset
        if not isinstance(y, type(None)):
            if isinstance(y, str):
                y_dataset = xarray.open_dataarray(y, engine = 'zarr', chunks = 'auto', mask_and_scale = False)
            else:
                y_dataset = y
            y_dataset = pm.persist(y_dataset)
            assert y_dataset.tiles == x_dataset.tiles
            if isinstance(classification, type(None)):
                classification = y_dataset.classification
            else:
                assert classification == y_dataset.classification
            if isinstance(y_nodata, type(None)):
                y_nodata = y_dataset.rio.nodata
            else:
                assert y_nodata == y_dataset.rio.nodata
            if isinstance(num_classes, type(None)):
                num_classes = y_dataset.num_classes
            else:
                assert num_classes == y_dataset.num_classes
        # Checking samples
        samples = []
        for i in range(len(x_dataset.names)):
            if names == 'all':
                samples.extend(x_dataset.samples[i])
            elif x_dataset.names[i] in names:
                samples.extend(x_dataset.samples[i])
        if not isinstance(y, type(None)):
            y_samples = []
            for i in range(len(y_dataset.names)):
                if names == 'all':
                    y_samples.extend(y_dataset.samples[i])
                if y_dataset.names[i] in names:
                    y_samples.extend(y_dataset.samples[i])
            assert y_samples == samples
        # Getting indices of samples
        indices = []
        for k, v in enumerate([x for xs in x_dataset.samples for x in xs]):
            if v in samples:
                indices.append(k)
        # Reading dataset
        for index in indices:
            if isinstance(x_stack, type(None)):
                x_stack = x_dataset[index].astype('float32').stack(data = ('y', 'x'))
            else:
                x_stack = xarray.concat([x_stack, x_dataset[index].astype('float32').stack(data = ('y', 'x'))], dim = 'data')
                x_stack = x_stack.transpose('data', 'band')
                x_stack = pm.persist(x_stack)
            if not isinstance(y, type(None)):
                if isinstance(y_stack, type(None)):
                    y_stack = y_dataset[index].stack(data = ('y', 'x'))
                else:
                    y_stack = xarray.concat([y_stack, y_dataset[index].stack(data = ('y', 'x'))], dim = 'data')
                y_stack = pm.persist(y_stack)
    if not isinstance(y, type(None)):
        return x_stack, y_stack, classification, y_nodata, num_classes, pm
    else:
        return x_stack, pm