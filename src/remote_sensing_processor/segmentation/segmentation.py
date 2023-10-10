import os
import collections
import numpy as np
import h5py
import joblib
import warnings

import sklearn.metrics

import torch
import torchmetrics
import transformers
import lightning as l

from remote_sensing_processor.segmentation.models import load_model, load_sklearn_model

# this warning usually appears on sanity check if a loaded tile is empty
warnings.filterwarnings("ignore", message="No positive samples in targets")
warnings.filterwarnings("ignore", message="exists and is not empty")
warnings.filterwarnings("ignore", message="could not find the monitored key in the returned metrics")
warnings.filterwarnings("ignore", message="Skipping val loop")

def segmentation_train(x_train, x_val, y_train, y_val, model, backbone, checkpoint, weights, model_file, epochs, batch_size, classification, num_classes, x_nodata, y_nodata, less_metrics, lr, multiprocessing):
    #checking if x and y have same number of datasets
    if not len(x_train) == len(y_train) or not len(x_val) == len(y_val):
        raise ValueError("Every x dataset must have a corresponding y dataset")
    #opening files and getting data shapes and metadata
    if isinstance(x_train[0], str):
        with h5py.File(x_train[0], 'r') as file:
            input_shape = file['data'].shape[2]
            input_dims = file['data'].shape[1]
            if x_nodata == None:
                x_nodata = file["data"].attrs['nodata']
    elif isinstance(x_train[0], np.ndarray):
        input_shape = x_train[0].shape[2]
        input_dims = x_train[0].shape[1]
    if isinstance(y_train[0], str):
        with h5py.File(y_train[0], 'r') as file:
            if classification == None:
                classification = file["data"].attrs['classification']
            if num_classes == None:
                if classification:
                    num_classes = file["data"].attrs['num_classes']
                else:
                    num_classes = 1
            if y_nodata == None:
                y_nodata = file["data"].attrs['nodata']
    elif isinstance(y_train[0], np.ndarray):
        if classification == None:
            classification = True
        if num_classes == None:
            if classification:
                num_classes = int(np.max(y_train[0]) + 1)
            else:
                num_classes = 1
    num_classes = int(num_classes)
    if model in ['BEiT', 'ConditionalDETR', 'Data2Vec', 'DETR', 'DPT', 'Mask2Former', 'MaskFormer', 'MobileNetV2', 'MobileViT', 'MobileViTV2', 'OneFormer', 'SegFormer', 'UperNet', 'DeepLabV3', 'FCN', 'LRASPP']:
        #deep learning pytorch models
        #checking if file extention is right
        if os.path.splitext(model_file)[1] != '.ckpt':
            raise ValueError("Wrong model file format: .ckpt file extention expected for " + model)
        #loading model
        if checkpoint != None:
            model = Model.load_from_checkpoint(checkpoint, input_shape = input_shape, input_dims = input_dims, num_classes = num_classes, classification = classification, y_nodata = y_nodata, lr = lr)
        else:
            model = Model(model, backbone, weights, input_shape, input_dims, num_classes, classification, y_nodata, less_metrics, lr)
        #setting up data sets generators
        datasets = []
        for i in range(len(x_train)):
            datasets.append(H5Dataset(x_train[i], y_train[i]))
        ds_train = torch.utils.data.ConcatDataset(datasets)
        if x_val != [None] and y_val != [None]:
            datasets = []
            for i in range(len(x_val)):
                datasets.append(H5Dataset(x_val[i], y_val[i]))
            ds_val = torch.utils.data.ConcatDataset(datasets)
        if multiprocessing:
            cpus = torch.multiprocessing.cpu_count()
        else:
            cpus = 0
        train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, pin_memory=True, num_workers = cpus)
        if x_val != [None] and y_val != [None]:
            val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, pin_memory=True, num_workers = cpus)
        #training
        checkpoint_callback = l.pytorch.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss" if x_val != [None] and y_val != [None] else "train_loss",
            mode="min",
            dirpath=os.path.dirname(model_file),
            filename=os.path.basename(os.path.splitext(model_file)[0]),
        )
        tb_logger = l.pytorch.loggers.TensorBoardLogger(save_dir=os.path.join(os.path.dirname(model_file), 'logs' , 'tensorboard'))
        csv_logger = l.pytorch.loggers.CSVLogger(save_dir=os.path.join(os.path.dirname(model_file), 'logs' , 'csv'))
        trainer = l.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], logger=[tb_logger, csv_logger])
        if x_val != [None] and y_val != [None]:
            trainer.fit(model, train_dataloader, val_dataloader)
        else:
            trainer.fit(model, train_dataloader)
    elif model in ["Nearest Neighbors", "Logistic Regression", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Gradient Boosting", "Multilayer Perceptron", "AdaBoost", "Naive Bayes", "QDA", "Ridge", "Lasso", "ElasticNet"]:
        #checking if file extention is right
        if os.path.splitext(model_file)[1] != '.joblib':
            raise ValueError("Wrong model file format: .joblib file extention expected for " + model)
        #loading train datasets
        x_train = sklearn_load_dataset(x_train, classification)
        y_train = sklearn_load_dataset(y_train, classification)
        #loading val datasets
        if x_val != [None] and y_val != [None]:
            x_val = sklearn_load_dataset(x_val, classification)
            y_val = sklearn_load_dataset(y_val, classification)
        if checkpoint != None:
            model = joblib.load(checkpoint)
            if model.model_name in ["Random Forest", "Gradient Boosting"]:
                model.model.n_estimators += 50
        else:
            model = SklearnModel(model, backbone, classification, epochs)
        model.fit(x_train, y_train)
        if not isinstance(x_val, list) and not isinstance(y_val, list):
            model.test(x_val, y_val)
        try:
            joblib.dump(model, model_file, compress=9)
        except:
            warnings.warn('Error while saving model, check if enough free space is available.')
    else:
        raise ValueError("Wrong model name. Check spelling or read a documentation and choose a supported model")
    return model

    
def segmentation_test(x_test, y_test, model, batch_size, multiprocessing):
    if not len(x_test) == len(y_test):
        raise ValueError("Every x dataset must have a corresponding y dataset")
    #loading model
    if isinstance(model, str):
        if '.ckpt' in model:
            model = Model.load_from_checkpoint(model)
        elif '.joblib' in model:
            model = joblib.load(model)
    if model.model_name in ['BEiT', 'ConditionalDETR', 'Data2Vec', 'DETR', 'DPT', 'Mask2Former', 'MaskFormer', 'MobileNetV2', 'MobileViT', 'MobileViTV2', 'OneFormer', 'SegFormer', 'UperNet', 'DeepLabV3', 'FCN', 'LRASPP']:
        #neural networks
        datasets = []
        for i in range(len(x_test)):
            datasets.append(H5Dataset(x_test[i], y_test[i]))
        ds_test = torch.utils.data.ConcatDataset(datasets)
        if multiprocessing:
            cpus = torch.multiprocessing.cpu_count()
        else:
            cpus = 0
        test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, pin_memory=True, num_workers = cpus)
        trainer = l.Trainer()
        trainer.test(model, dataloaders=test_dataloader)
    elif model.model_name in ["Nearest Neighbors", "Logistic Regression", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Gradient Boosting", "Multilayer Perceptron", "AdaBoost", "Naive Bayes", "QDA", "Ridge", "Lasso", "ElasticNet"]:
        #sklearn models
        classification = model.classification
        #loading test datasets
        x_test = sklearn_load_dataset(x_test, classification)
        y_test = sklearn_load_dataset(y_test, classification)
        model.test(x_test, y_test)
    else:
        raise ValueError("Wrong model name. Check spelling or read a documentation and choose a supported model")
    

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        #self.classification = classification
        #self.x_nodata = x_nodata
        #self.y_nodata = y_nodata
        if isinstance(x, str):
            #dataset will be opened in getitem
            self.x_file_path = x
            self.x_dataset = None
            with h5py.File(self.x_file_path, 'r') as file:
                self.dataset_len = len(file["data"])
                #if x_nodata == None:
                    #self.x_nodata = file["data"].attrs['nodata']
        elif isinstance(x, np.ndarray):
            self.x_dataset = x
            self.dataset_len = len(x)
        if isinstance(y, str):
            self.y_file_path = y
            self.y_dataset = None
            #with h5py.File(self.y_file_path, 'r') as file:
                #self.categorical = file["data"].attrs['categorical']
                #if y_nodata == None:
                    #self.y_nodata = file["data"].attrs['nodata']
                    #self.nfn = file["data"].attrs['nfn']
        elif isinstance(y, np.ndarray):
            self.y_dataset = y
        #if self.categorical and self.y_nodata != None:
            #self.y_nodata = int(self.y_nodata)

    def __getitem__(self, index):
        if self.x_dataset is None and self.x_file_path != None:
            self.x_dataset = h5py.File(self.x_file_path, 'r')["data"]
        if self.y_dataset is None and self.y_file_path != None:
            self.y_dataset = h5py.File(self.y_file_path, 'r')["data"]
        x = self.x_dataset[index]
        #x = (x / 15000)
        y = self.y_dataset[index]
        #y = (y + 1)/(1+1)
        """if self.nfn and self.x_nodata != None and self.y_nodata != None:
            if self.categorical == True:
                x = np.where(np.broadcast_to(y[self.y_nodata], x.shape) == 1, self.x_nodata, x)  
                y[self.y_nodata] = np.where(np.broadcast_to(x[0], y[self.y_nodata].shape) == self.x_nodata, 1, 0)
            else:
                x = np.where(np.broadcast_to(y, x.shape) == self.y_nodata, self.x_nodata, x)
                y = np.where(np.broadcast_to(x[0], y.shape) == self.x_nodata, self.y_nodata, y)
        if self.classification:
            y = np.int64(y)
        else:
            y = np.float32(y)"""
        return np.float32(x), y

    def __len__(self):
        return self.dataset_len


class PredDataset(torch.utils.data.Dataset):
    #same as H5Dataset, but for x only
    def __init__(self, x):
        if isinstance(x, str):
            self.x_file_path = x
            self.x_dataset = None
            with h5py.File(self.x_file_path, 'r') as file:
                self.dataset_len = len(file["data"])
        elif isinstance(x, np.ndarray):
            self.x_dataset = x
            self.dataset_len = len(x)

    def __getitem__(self, index):
        if self.x_dataset is None and self.x_file_path != None:
            self.x_dataset = h5py.File(self.x_file_path, 'r')["data"]
        x = self.x_dataset[index]
        return np.float32(x)

    def __len__(self):
        return self.dataset_len


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
    
    def forward(self, x):
        if isinstance(self.model, transformers.OneFormerForUniversalSegmentation):
            #oneformer also requires tokenized tasks as inputs, task is semantic
            t = torch.tensor([[49406,   518, 10549,   533, 29119,  1550, 49407,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0]], device = self.device)
            t = torch.cat([t]*x.shape[0], dim=0)
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar = True)
        if self.classification:
            #commented metrics somehow make detr freeze after first steps, hope this is temporary fix
            if self.less_metrics:
                self.log('train_acc', torchmetrics.functional.classification.accuracy(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                #self.log('train_precision', torchmetrics.functional.precision(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                #self.log('train_recall', torchmetrics.functional.recall(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('train_auroc', torchmetrics.functional.auroc(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                #self.log('train_iou', torchmetrics.functional.jaccard_index(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
            else:
                self.log('train_acc', torchmetrics.functional.classification.accuracy(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                self.log('train_precision', torchmetrics.functional.precision(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('train_recall', torchmetrics.functional.recall(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('train_auroc', torchmetrics.functional.auroc(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                self.log('train_iou', torchmetrics.functional.jaccard_index(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
        else:
            self.log('train_r2', torchmetrics.functional.r2_score(torch.reshape(torch.squeeze(pred), (pred.shape[0], -1)), torch.reshape(y, (y.shape[0], -1))), on_step=True, on_epoch=True, prog_bar = True)
            self.log('train_kendall', torchmetrics.functional.kendall_rank_corrcoef(torch.reshape(torch.squeeze(pred), (pred.shape[0], -1)), torch.reshape(y, (y.shape[0], -1))).mean(), on_step=True, on_epoch=True, prog_bar = True)
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
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar = True)
        if self.classification:
            if self.less_metrics:
                self.log('val_acc', torchmetrics.functional.classification.accuracy(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                #self.log('val_precision', torchmetrics.functional.precision(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                #self.log('val_recall', torchmetrics.functional.recall(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('val_auroc', torchmetrics.functional.auroc(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                #self.log('val_iou', torchmetrics.functional.jaccard_index(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
            else:
                self.log('val_acc', torchmetrics.functional.classification.accuracy(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                self.log('val_precision', torchmetrics.functional.precision(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('val_recall', torchmetrics.functional.recall(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('val_auroc', torchmetrics.functional.auroc(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                self.log('val_iou', torchmetrics.functional.jaccard_index(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
        else:
            self.log('val_r2', torchmetrics.functional.r2_score(torch.reshape(torch.squeeze(pred), (pred.shape[0], -1)), torch.reshape(y, (y.shape[0], -1))), on_step=True, on_epoch=True, prog_bar = True)
            self.log('val_kendall', torchmetrics.functional.kendall_rank_corrcoef(torch.reshape(torch.squeeze(pred), (pred.shape[0], -1)), torch.reshape(y, (y.shape[0], -1))).mean(), on_step=True, on_epoch=True, prog_bar = True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        y = self.ytype(y)
        pred = self.process_output(pred, x)
        if self.classification:
            loss = self.loss_fn(pred, y)
        else:
            loss = self.mse_loss(pred.squeeze(), y.squeeze(), ignore_index = self.y_nodata)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar = True)
        if self.classification:
            # commented metrics somehow make detr freeze after first steps, hope this is temporary fix
            if self.less_metrics:
                self.log('test_acc', torchmetrics.functional.classification.accuracy(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                #self.log('test_precision', torchmetrics.functional.precision(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                #self.log('test_recall', torchmetrics.functional.recall(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('test_auroc', torchmetrics.functional.auroc(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                #self.log('test_iou', torchmetrics.functional.jaccard_index(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
            else:
                self.log('test_acc', torchmetrics.functional.classification.accuracy(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                self.log('test_precision', torchmetrics.functional.precision(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('test_recall', torchmetrics.functional.recall(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True)
                self.log('test_auroc', torchmetrics.functional.auroc(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
                self.log('test_iou', torchmetrics.functional.jaccard_index(pred, y, 'multiclass', num_classes = self.num_classes), on_step=True, on_epoch=True, prog_bar = True)
        else:
            self.log('test_r2', torchmetrics.functional.r2_score(torch.reshape(torch.squeeze(pred), (pred.shape[0], -1)), torch.reshape(y, (y.shape[0], -1))), on_step=True, on_epoch=True, prog_bar = True)
            self.log('test_kendall', torchmetrics.functional.kendall_rank_corrcoef(torch.reshape(torch.squeeze(pred), (pred.shape[0], -1)), torch.reshape(y, (y.shape[0], -1))).mean(), on_step=True, on_epoch=True, prog_bar = True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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
                    size=x.shape[2:4],
                    mode="bilinear",
                    align_corners=False,
                )
        elif isinstance(pred, transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrSegmentationOutput) or isinstance(pred, transformers.models.detr.modeling_detr.DetrSegmentationOutput):
            class_queries_logits = pred.logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = pred.pred_masks  # [batch_size, num_queries, height, width]
            
            # Remove the null class `[..., :-1]`
            # Detr treats 0 as separate class, conditionaldetr treats it as nullclass
            if isinstance(pred, transformers.models.detr.modeling_detr.DetrSegmentationOutput):
                masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
            else:
                masks_classes = class_queries_logits.softmax(dim=-1)
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

            if pred.shape[2:4] != x.shape[2:4]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=x.shape[2:4],
                    mode="bilinear",
                    align_corners=False,
                )
        elif isinstance(pred, transformers.models.mask2former.modeling_mask2former.Mask2FormerForUniversalSegmentationOutput) or isinstance(pred, transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput):
            class_queries_logits = pred.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = pred.masks_queries_logits  # [batch_size, num_queries, height, width]
            masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]
            
            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            if pred.shape[2:4] != x.shape[2:4]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=x.shape[2:4],
                    mode="bilinear",
                    align_corners=False,
                )
        elif isinstance(pred, transformers.models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput):
            class_queries_logits = pred.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = pred.masks_queries_logits  # [batch_size, num_queries, height, width]
            masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            if pred.shape[2:4] != x.shape[2:4]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=x.shape[2:4],
                    mode="bilinear",
                    align_corners=False,
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
        out = (pred[~mask]-target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
        
class SklearnModel:
    def __init__(self, model, backbone, classification, epochs):
        self.classification = classification
        self.model_name = model
        self.model = load_sklearn_model(model, backbone, classification, epochs)
    
    def fit(self, x, y):
        self.model.fit(x, y)
    
    def test(self, x, y):
        pred = self.predict(x)
        #print(np.unique(pred))
        if self.classification:
            print('Accuracy: ', sklearn.metrics.accuracy_score(y.flatten(), pred.flatten()))
            print('Precision: ', sklearn.metrics.precision_score(y.flatten(), pred.flatten(), average = 'macro', zero_division = 1))
            print('Recall: ', sklearn.metrics.recall_score(y.flatten(), pred.flatten(), average = 'macro', zero_division = 1))
            #print('ROC_AUC: ', sklearn.metrics.roc_auc_score(y.flatten(), pred.flatten(), average = 'micro', multi_class = 'ovr'))
            print('IOU: ', sklearn.metrics.jaccard_score(y.flatten(), pred.flatten(), average = 'micro', zero_division = 1))
        else:
            print('R2: ', sklearn.metrics.r2_score(y, pred))
            print('MSE: ', sklearn.metrics.mean_squared_error(y, pred))
    
    def predict(self, x):
        pred = self.model.predict(x)
        return pred
        
    
def sklearn_load_dataset(ds, classification):
    if isinstance(ds[0], str):
        datasets = []
        for d in ds:
            with h5py.File(d, 'r') as file:
                d = file['data']
                d = d[...]
                datasets.append(d)
    else:
        datasets = ds
    #concatenate datasets
    ds = np.concatenate(datasets, axis = 0)
    #if x dataset
    if len(ds.shape) == 4:
        #stack all tiles to single image
        ds = np.hstack([i for i in ds])
        ds = ds.reshape(ds.shape[0], -1)
        ds = np.moveaxis(ds, 0, -1)
    #if y dataset
    elif len(ds.shape) == 3:
        #stack all tiles to single image
        ds = np.vstack([i for i in ds])
        ds = ds.reshape(-1)
    return ds