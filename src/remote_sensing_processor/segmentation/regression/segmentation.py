"""Regression main functionality."""

from pydantic import BaseModel, InstanceOf, NonNegativeInt, PositiveInt, TypeAdapter, validate_call
from typing import Any, Literal, Optional, Union

import warnings
from pathlib import Path

import joblib

import numpy as np

import datasets
import lightning as l
import torch
from torchvision import tv_tensors

from sklearn.base import BaseEstimator

from remote_sensing_processor.common.torch_test import cuda_test
from remote_sensing_processor.common.types import (
    FilePath,
    ListOfDict,
    ListOfStr,
    LoadRSPDS,
    NewPath,
    SingleOrList,
    SKLModel,
    TorchLoss,
    TorchNNModel,
    TorchTransform,
)
from remote_sensing_processor.segmentation.regression.losses import setup_loss
from remote_sensing_processor.segmentation.regression.metrics import RegresssionMetrics
from remote_sensing_processor.segmentation.regression.models import (
    RegressionModels,
    pytorch_models,
    sklearn_models,
)
from remote_sensing_processor.segmentation.segmentation import (
    DataModule,
    Dataset,
    Model,
    SklearnModel,
    setup_trainer,
    sklearn_load_dataset,
)
from remote_sensing_processor.segmentation.tiles import replace_y_in_meta


# These warnings usually appear on sanity check if a loaded tile is empty
warnings.filterwarnings("ignore", message=".*No positive samples in targets.*")
warnings.filterwarnings("ignore", message=".*exists and is not empty.*")
warnings.filterwarnings("ignore", message=".*could not find the monitored key in the returned metrics.*")
warnings.filterwarnings("ignore", message=".*Skipping val loop.*")


class DS(BaseModel):
    """Dataset class for user input."""

    path: LoadRSPDS
    sub: Union[Literal["all"], ListOfStr]
    y: Optional[str] = None
    predict: Optional[bool] = False


ListOfDS = SingleOrList[DS]


class RegressionDataset(Dataset):
    """Semantic segmentation dataset."""

    def __init__(self, dataset: DS) -> None:
        super().__init__(dataset)

        # Change detection dataset have bi-temporal x variable
        if self.meta["task"] in ["change_detection"]:
            raise ValueError("dataset is not a semantic dataset")

        if "y" in self.meta:
            # Needed only if y exists, because x is always the same
            if self.meta["task"] != "regression":
                raise ValueError("dataset is not a regression dataset")
            self.meta = replace_y_in_meta(self.meta, dataset)

        # Setting up regression specific parameters
        self.y_nodata = self.meta["y"]["nodata"] if "y" in self.meta else None
        self.y_dtype = self.meta["y"]["dtype"] if "y" in self.meta else None


class RegressionDataModule(DataModule):
    """Data module for regression."""

    def __init__(
        self,
        train_datasets: Optional[ListOfDict] = None,
        val_datasets: Optional[ListOfDict] = None,
        test_datasets: Optional[ListOfDict] = None,
        pred_dataset: Optional[dict] = None,
        repeat: Optional[int] = 1,
        augment: Optional[Union[bool, tuple[Union[str, TorchTransform], ...]]] = False,
        batch_size: Optional[int] = 32,
        num_workers: Optional[Union[int, Literal["auto"]]] = "auto",
    ) -> None:
        if train_datasets is not None:
            train_datasets = TypeAdapter(ListOfDS).validate_python(train_datasets)
        if val_datasets is not None:
            val_datasets = TypeAdapter(ListOfDS).validate_python(val_datasets)
        if test_datasets is not None:
            test_datasets = TypeAdapter(ListOfDS).validate_python(test_datasets)
        if pred_dataset is not None:
            pred_dataset = TypeAdapter(DS).validate_python(pred_dataset)

        super().__init__(
            train_datasets,
            val_datasets,
            test_datasets,
            pred_dataset,
            repeat,
            augment,
            batch_size,
            num_workers,
        )

    def dataset_check(self, dataset: ListOfDS) -> list[RegressionDataset]:
        """Check and preprocess datasets."""
        final_ds = []
        for ds in dataset:
            final_ds.append(RegressionDataset(ds))

        # As reference is only needed for prediction, there can not be more than one dataset
        self.reference = final_ds[0].reference

        # Checking data values
        self.assert_common(final_ds)

        if not all(d.y_nodata == final_ds[0].y_nodata for d in final_ds):
            raise ValueError("y nodatas of input datasets are different")
        self.y_nodata = final_ds[0].y_nodata

        if not all(d.y_dtype == final_ds[0].y_dtype for d in final_ds):
            raise ValueError("y_dtypes of input datasets are different")
        self.y_dtype = final_ds[0].y_dtype

        return final_ds

    def setup_datasets(self, ds: list[RegressionDataset]) -> datasets.Dataset:
        """Set up a dataset for one stage."""
        ds_processed = []
        for d in ds:
            for file in d.files:
                # Loading the dataset
                dataset = datasets.Dataset.load_from_disk(file.as_posix())
                dataset.set_format("torch")

                # Filtering columns
                columns = ["key", "x"]
                if "y" in d.meta:
                    dataset = dataset.rename_column("y_" + d.meta["y"]["name"], "y")
                    columns.append("y")
                dataset = dataset.select_columns(columns)

                ds_processed.append(dataset)

        # Concatenating datasets
        return datasets.concatenate_datasets(ds_processed * self.repeat)

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: Optional[int]) -> dict:
        """Apply augmentations to data."""
        if self.trainer and self.trainer.training and self.transform is not None:
            batch["x"], batch["y"] = self.transform(tv_tensors.Image(batch["x"]), tv_tensors.Mask(batch["y"]))
        return batch


class RegressionModel(Model, RegressionModels, RegresssionMetrics):
    """RSP Torch-based model for regression."""

    def __init__(
        self,
        model: Union[str, TorchNNModel],
        backbone: Optional[str],
        weights: Optional[str],
        loss: Optional[Union[str, TorchLoss]],
        metrics: Optional[ListOfDict],
        input_shape: int,
        input_dims: int,
        y_nodata: Optional[Union[int, float]],
        lr: Optional[float],
        precision: Optional[str],
        val: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            input_shape=input_shape,
            input_dims=input_dims,
            y_nodata=y_nodata,
            lr=lr,
            precision=precision,
            scheduler_opt="val_loss" if val else "train_loss",
            overwrite_loss=loss is not None,
        )

        if isinstance(model, str):
            self.model = self.load_model(model_name=model, bb=backbone, weights=weights, **kwargs)
        else:
            self.model = self.validate_model(model=model)

        self.setup_metrics(metrics)
        self.loss_fn = setup_loss(loss=loss)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Here data is being fit to the model."""
        key = batch["key"]
        y = batch.get("y")

        pred, loss = self.model(batch)

        # Calculate loss
        if y is not None and (loss is None or self.overwrite_loss):
            # Removing nodata
            y_filtered = y.reshape(-1)
            pred_filtered = pred.reshape(-1)
            valid_mask = y_filtered != self.y_nodata
            y_filtered = y_filtered[valid_mask]
            pred_filtered = pred_filtered[valid_mask]
            # Calculating loss
            loss = self.loss_fn(pred_filtered, y_filtered.float())
        return y, pred, loss, key

    def post_process_predict(self, pred: torch.Tensor) -> torch.Tensor:
        """Prepare predictions to mapping."""
        return pred

    def log_all(self, y: torch.Tensor, pred: torch.Tensor, loss: torch.Tensor, stage: str) -> None:
        """Logging all the metrics."""
        if y is not None:
            # Removing nodata
            y = y.reshape(-1)
            pred = pred.reshape(-1)
            valid_mask = y != self.y_nodata
            y = y[valid_mask]
            pred = pred[valid_mask]
            if loss is not None:
                self.log(stage + "_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            if stage == "train":
                if self.metrics_epoch_train is not None:
                    self.log_dict(self.metrics_epoch_train(pred, y), on_epoch=True, on_step=False, prog_bar=False)
                if self.metrics_step_train is not None:
                    self.log_dict(self.metrics_step_train(pred, y), on_epoch=True, on_step=True, prog_bar=False)
                if self.metrics_verbose_train is not None:
                    self.log_dict(self.metrics_verbose_train(pred, y), on_epoch=True, on_step=True, prog_bar=True)
            if stage == "val":
                if self.metrics_epoch_val is not None:
                    self.log_dict(self.metrics_epoch_val(pred, y), on_epoch=True, on_step=False, prog_bar=False)
                if self.metrics_step_val is not None:
                    self.log_dict(self.metrics_step_val(pred, y), on_epoch=True, on_step=True, prog_bar=False)
                if self.metrics_verbose_val is not None:
                    self.log_dict(self.metrics_verbose_val(pred, y), on_epoch=True, on_step=True, prog_bar=True)
            if stage == "test":
                if self.metrics_epoch_test is not None:
                    self.log_dict(self.metrics_epoch_test(pred, y), on_epoch=True, on_step=False, prog_bar=False)
                if self.metrics_step_test is not None:
                    self.log_dict(self.metrics_step_test(pred, y), on_epoch=True, on_step=True, prog_bar=False)
                if self.metrics_verbose_test is not None:
                    self.log_dict(self.metrics_verbose_test(pred, y), on_epoch=True, on_step=True, prog_bar=True)


class SklearnRegressionModel(SklearnModel, RegressionModels, RegresssionMetrics):
    """RSP Sklearn-based model for regression."""

    def __init__(
        self,
        model: Union[str, SKLModel],
        backbone: Optional[str],
        metrics: Optional[ListOfDict],
        y_nodata: Optional[Union[int, float]],
        generate_features: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            generate_features=generate_features,
            y_nodata=y_nodata,
        )
        if isinstance(model, str):
            self.model = self.load_sklearn_model(model_name=model, bb=backbone, **kwargs)
        else:
            self.model = model
        self.init_metrics(metrics)

    def init_metrics(self, metrics: Optional[ListOfDict]) -> None:
        """Initialize metrics."""
        self.setup_metrics(metrics, sklearn=True)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        # noinspection PyUnresolvedReferences
        self.model.fit(x, y)
        self.test(x, y)

    def test(self, x: np.ndarray, y: np.ndarray) -> None:
        """Test the model."""
        # noinspection PyUnresolvedReferences
        pred = self.model.predict(x)

        # Calculating and printing metrics
        filtered = np.where(y != self.y_nodata, True, False).nonzero()[0].tolist()
        # noinspection PyUnresolvedReferences
        metrics = self.metrics(torch.tensor(pred[filtered]).float(), torch.tensor(y[filtered]).float())
        for metric, val in metrics.items():
            print(metric, val.item())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict with the model."""
        # noinspection PyUnresolvedReferences
        return self.model.predict(x)

    def setup_warm_start(self, **kwargs: Any) -> None:
        """Configure models to support warm_start."""
        self.set_warm_start(**kwargs)


@validate_call
def train(
    train_datasets: ListOfDict,
    val_datasets: Union[ListOfDict, None],
    model_file: NewPath,
    model: Union[str, TorchNNModel, SKLModel],
    backbone: Optional[str] = None,
    checkpoint: Optional[FilePath] = None,
    weights: Optional[str] = None,
    epochs: Optional[dict[str, Any]] = None,
    loss: Optional[Union[str, TorchLoss]] = None,
    metrics: Optional[ListOfDict] = None,
    batch_size: Optional[PositiveInt] = 32,
    repeat: Optional[PositiveInt] = 1,
    augment: Optional[Union[bool, tuple[Union[str, TorchTransform], ...]]] = False,
    lr: Optional[float] = 1e-3,
    generate_features: Optional[bool] = False,
    num_workers: Optional[Union[NonNegativeInt, Literal["auto"]]] = 0,
    precision: Optional[str] = None,
    **kwargs: Any,
) -> Union[RegressionModel, SklearnRegressionModel]:
    """
    Trains segmentation model.

    Parameters
    ----------
    train_datasets : dict or list of dicts
        Dataset generated by generate_tiles() function that will be used to train the model.
        Each dataset can contain 3 elements:
        `path`: a path to a dataset. Required parameter.
        `sub`: subdataset name, list of subdataset names or 'all'. Required parameter.
        `y`: if there is more than one target variable in dataset,
        then the name of the variable that should be used for training should be defined. Optional parameter.
        You can provide a list of datasets to train model on multiple datasets.
    val_datasets : dict or list of dicts or None
        Dataset generated by generate_tiles() function that will be used to validate the model.
        Each dataset can contain 3 elements:
        `path`: a path to a dataset. Required parameter.
        `sub`: subdataset name, list of subdataset names or 'all'. Required parameter.
        `y`: if there is more than one target variable in dataset,
        then the name of the variable that should be used for validation should be defined. Optional parameter.
        You can provide a list of datasets to validate model on multiple datasets.
        Can be set to None if no validation is needed.
    model_file : path as a string
        Checkpoint file where model will be saved after training.
        File extension must be *.ckpt for neural networks and *.joblib for scikit-learn models.
    model : str or torch.nn or sklearn model
        Name of model architecture, pytorch regression model or sklearn regression model.
    backbone : str (optional)
        Backbone, solver or kernel of a model, if multiple backbones are supported.
    checkpoint : path as a string (optional)
        Checkpoint file (*.ckpt or *.joblib) of a pre-trained model to fine-tune.
    weights : str (optional)
        Name of pre-trained weights to fine-tune. Only works for neural networks.
    epochs : dict (optional)
        Dict of values that set the number of training epochs and early stopping parameter for Deep Learning models.
        `max_epochs` (int): the maximum number of epochs.
        `early_stopping` (bool): is early stopping enabled.
        `min_delta` (float): minimum change in the monitored quantity to qualify as an improvement. Optional parameter.
        `patience` (int): number of epochs with no improvement after which training will be stopped. Optional parameter.
        If you only want to initialize model for future testing or prediction, set `max_epochs` to 0.
        If not set, will use max_epochs = 5 and early_stopping with default parameters.
        `epochs` have no effect for Scikit-Learn models.
        Please, set `num_iter`, `tol` and other epochs-related parameters via **kwargs.
    loss : str or torch.nn (optional)
        Loss function that will be used during the training.
        The default one is MSE or default loss for HuggingFace Transformers models.
        You can use any custom loss function, but it must inherit `torch.nn.modules.loss._Loss`.
    metrics : dict or list of dicts (optional)
        Metrics that will be used to evaluate model performance and logged.
        Can be a single dict or list of dicts. Each dict corresponds to one metric.
        `name` (str): name of a metric. If name is one of supported metrics, it will be automatically loaded and used.
        `log` (str): logging levels can be 'epoch' - to log the metric only on the end of each epoch,
        'step' - to log on each training step and 'verbose' - to log on each step and show alongside progress bar.
        `metric` (Metric): your custom metric object. Optional parameter.
        You can use any custom metrics, but they must inherit `torchmetrics.metric`.
        If not set, accuracy and mean IoU are verbose logged and precision and recall are logged after each epoch.
    batch_size : int (default = 32)
        Number of training samples used in one iteration. Only works for neural networks.
    repeat : int (default = 1)
        Increase size of a dataset by repeating it n times. Can be useful if dataset is very small.
    augment : bool or sequence of str (default = False)
        Apply augmentations to dataset. Only works for neural networks. No augmentations applied if set to False.
        If set to True then the default augmentations (`RandomResizedCrop`, `RandomHorizontalFlip`) are applied.
        You can pass your own sequence of augmentations, they will be applied to data in the given order.
        You can use any custom augmentations, but they must inherit `torchvision.transforms.v2.Transform`.
    lr : float (default = 1e-3)
        Learning rate of a model. Lower value results usually in better model convergence, but much slower training.
        `lr` have no effect for Scikit-Learn models.
        Please, set `learning_rate_init`, `alpha` and other lr-related parameters via **kwargs.
    generate_features : bool (default = False)
        If set to True, intensity, gradient intensity and local structure features will be generated, as described
        `here <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html>`_.
        Can result in better segmentation quality, but can also significantly increase training time.
        Only works for scikit-learn models.
    num_workers : int or 'auto' (default = 0)
        Number of parallel workers that will load the data.
        Set 'auto' to let RSP choose the optimal number of workers, set 0 to disable multiprocessing.
        Can increase training speed, but can also cause errors (e.g. pickling errors).
    precision : str (optional)
        Precision that will be used in training process.
        Lower precision requires less memory, but can sometimes cause errors.
        More info can be found `here <https://lightning.ai/docs/pytorch/stable/common/precision.html>`_
    **kwargs
        Additional keyword arguments that are used to initialize model.
        They are different for every model, so read the documentation.

    Returns
    -------
    torch.nn model or SklearnModel
        Trained model.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> x = ["/home/rsp_test/mosaics/sentinel/", "/home/rsp_test/mosaics/dem/dem.tif"]
        >>> y = [
        ...     {"name": "nitrogen", "path": "/home/rsp_test/mosaics/nitrogen.tif"},
        ...     {"name": "phosphorus", "path": "/home/rsp_test/mosaics/phosphorus.tif"},
        ... ]
        >>> out_file = "/home/rsp_test/model/chem_dataset.rspds"
        >>> dataset_path = rsp.regression.generate_tiles(
        ...     x,
        ...     y,
        ...     out_file,
        ...     tile_size=256,
        ...     shuffle=True,
        ...     split={"train": 3, "val": 1, "test": 1},
        ... )
        >>> # We will train model to predict nitrogen content
        >>> train_ds = {"path": dataset_path, "sub": "train", "y": "nitrogen"}
        >>> val_ds = {"path": dataset_path, "sub": "val", "y": "nitrogen"}
        >>> model = rsp.regression.train(
        ...     train_ds,
        ...     val_ds,
        ...     model="UperNet",
        ...     backbone="ConvNeXTV2",
        ...     model_file="/home/rsp_test/model/upernet.ckpt",
        ...     epochs={"max_epochs": 100, "early_stopping": False},
        ...     batch_size=32,
        ... )
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
        223/223 [1:56:20<00:00, 31.30s/it, v_num=54,
        train_loss_step=0.326, train_acc_step=0.871, train_auroc_step=0.796, train_iou_step=0.655,
        val_loss_step=0.324, val_acc_step=0.869, val_auroc_step=0.620, val_iou_step=0.678,
        val_loss_epoch=0.334, val_acc_epoch=0.807, val_auroc_epoch=0.795, val_iou_epoch=0.688,
        train_loss_epoch=0.349, train_acc_epoch=0.842, train_auroc_epoch=0.797, train_iou_epoch=0.648]
        `Trainer.fit` stopped: `max_epochs=10` reached.

        >>> ds_mo = "/home/rsp_test/model/montana.rspds"
        >>> ds_id = "/home/rsp_test/model/idaho.rspds"
        >>> # Training on two different datasets - one from Montana and one from Idaho
        >>> train_ds = [
        ...     {"path": ds_mo, "sub": ["area_1", "area_2"]},
        ...     {"path": ds_id, "sub": ["area_3", "area_6", "area8"]},
        ... ]
        >>> val_ds = [
        ...     {{'path': ds_mo, 'sub': ['area_3', 'area_4']},
        ...     {'path': ds_id, 'sub': ['area_1']},
        ... ]
        >>> model = rsp.regression.train(
        ...     train_ds,
        ...     val_ds,
        ...     model="UperNet",
        ...     backbone="ConvNeXTV2",
        ...     model_file="/home/rsp_test/model/upernet.ckpt",
        ...     epochs={"max_epochs": 100, "early_stopping": False},
        ...     batch_size=32,
        ... )
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
        223/223 [1:56:20<00:00, 31.30s/it, v_num=54, train_loss_step=0.326,
        train_acc_step=0.871, train_auroc_step=0.796, train_iou_step=0.655,
        val_loss_step=0.324, val_acc_step=0.869, val_auroc_step=0.620, val_iou_step=0.678,
        val_loss_epoch=0.334, val_acc_epoch=0.807, val_auroc_epoch=0.795, val_iou_epoch=0.688,
        train_loss_epoch=0.349, train_acc_epoch=0.842, train_auroc_epoch=0.797, train_iou_epoch=0.648]
        `Trainer.fit` stopped: `max_epochs=100` reached.
    """
    # Setting up data module
    dm = RegressionDataModule(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        repeat=repeat,
        augment=augment,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Deep learning pytorch models
    if model in pytorch_models or isinstance(model, torch.nn.Module):
        if not cuda_test():
            warnings.warn("CUDA or MPS is not available. Training on CPU could be very slow.", stacklevel=1)
        # Checking if file extension is right
        if ".ckpt" not in model_file.suffixes:
            raise ValueError("Wrong model file format: .ckpt file extention expected.")

        # Loading model
        if checkpoint is not None:
            model = RegressionModel.load_from_checkpoint(
                checkpoint,
                loss=loss,
                metrics=metrics,
                input_shape=dm.input_shape,
                input_dims=dm.input_dims,
                y_nodata=dm.y_nodata,
                lr=lr,
                val=val_datasets is not None,
            )
        else:
            model = RegressionModel(
                model=model,
                backbone=backbone,
                weights=weights,
                loss=loss,
                metrics=metrics,
                input_shape=dm.input_shape,
                input_dims=dm.input_dims,
                y_nodata=dm.y_nodata,
                lr=lr,
                precision=precision,
                val=val_datasets is not None,
                **kwargs,
            )
        # if epochs == 0 - only initialize model without training
        if epochs is not None and epochs["max_epochs"] == 0:
            model.save_checkpoint(model_file)
        else:
            # Setting up trainer
            trainer = setup_trainer(model_file, epochs, val_datasets is not None, precision)
            # Training
            trainer.fit(model, dm)
    # Sklearn ML models
    elif model in sklearn_models or isinstance(model, BaseEstimator):
        dm.setup(stage="fit")
        # Checking if file extension is right
        if ".joblib" not in model_file.suffixes:
            raise ValueError("Wrong model file format: .joblib file extension expected.")
        # Loading train datasets
        x_train, y_train, _ = sklearn_load_dataset(dm, "train", generate_features)
        if checkpoint is not None:
            model = joblib.load(checkpoint)
            model.setup_warm_start(**kwargs)
        else:
            model = SklearnRegressionModel(
                model=model,
                backbone=backbone,
                metrics=metrics,
                y_nodata=dm.y_nodata,
                generate_features=generate_features,
                **kwargs,
            )
        print("Training")
        model.fit(x_train, y_train)
        del x_train
        del y_train
        # Validation
        if val_datasets is not None:
            print("Validation")
            x_val, y_val, _ = sklearn_load_dataset(dm, "val", generate_features)
            model.test(x_val, y_val)
        try:
            joblib.dump(model, model_file, compress=9)
        except Exception:
            warnings.warn("Error while saving model, check if enough free space is available.", stacklevel=1)
    else:
        raise ValueError("Wrong model name. Check spelling or read a documentation and choose a supported model")
    return model


@validate_call
def test(
    test_datasets: ListOfDict,
    model: Union[FilePath, InstanceOf[RegressionModel], InstanceOf[SklearnRegressionModel]],
    metrics: Optional[ListOfDict] = None,
    batch_size: Optional[PositiveInt] = 32,
    num_workers: Optional[Union[NonNegativeInt, Literal["auto"]]] = 0,
) -> None:
    """
    Tests segmentation model.

    Parameters
    ----------
    test_datasets : dict or list of dicts
        Dataset generated by generate_tiles() function that will be used to test the model.
        Each dataset can contain 3 elements:
        `path` (path as str): a path to a dataset. Required parameter.
        `sub` (str): subdataset name, list of subdataset names or 'all'. Required parameter.
        `y` (str): if there is more than one target variable in dataset,
        then the name of the variable that should be used for testing should be defined. Optional parameter.
        You can provide a list of datasets to test model on multiple datasets.
    model : torch.nn model or SklearnModel or path to a model file
        Model to test. You can pass the model object returned by `train()` function
        or file (*.ckpt or *.joblib) where model is stored.
    metrics : dict or list of dicts (optional)
        Metrics that will be used to evaluate model performance and logged.
        Can be a single dict or list of dicts. Each dict corresponds to one metric.
        `name` (str): name of a metric. If name is one of supported metrics, it will be automatically loaded and used.
        `log` (str): logging levels can be 'epoch' - to log the metric only on the end of each epoch,
        'step' - to log on each training step and 'verbose' - to log on each step and show alongside progress bar.
        `metric` (Metric): your custom metric object. Optional parameter.
        You can use any custom metrics, but they must inherit `torchmetrics.metric`.
        If not set, will evaluate the metrics used in training process.
    batch_size : int (default = 32)
        Number of samples used in one iteration. Only works for neural networks.
    num_workers: int or 'auto' (default = 0)
        Number of parallel workers that will load the data.
        Set 'auto' to let RSP choose the optimal number of workers, set 0 to disable multiprocessing.
        Can increase training speed, but can also cause errors (e.g. pickling errors).

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> x, y, out_file = ...
        >>> ds = rsp.regression.generate_tiles(
        ...     x,
        ...     y,
        ...     out_file,
        ...     tile_size=256,
        ...     shuffle=True,
        ...     split={"train": 3, "val": 1, "test": 1},
        ... )
        >>> model = rsp.regression.train(
        ...     {"path": ds, "sub": "train"},
        ...     {"path": ds, "sub": "val"},
        ...     model="UperNet",
        ...     backbone="ConvNeXTV2",
        ...     model_file="/home/rsp_test/model/upernet.ckpt",
        ...     batch_size=32,
        ... )
        >>> rsp.regression.test({"path": ds, "sub": "test"}, model=model, batch_size=32)
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
    # Loading model
    if isinstance(model, Path):
        if ".ckpt" in model.suffixes:
            model = RegressionModel.load_from_checkpoint(model)
        elif ".joblib" in model.suffixes:
            model = joblib.load(model)
        else:
            raise ValueError("Wrong model extension. Should be .ckpt or .joblib")
    # Re-initialize metrics if needed
    if metrics is not None:
        model.init_metrics(metrics)
    dm = RegressionDataModule(test_datasets=test_datasets, batch_size=batch_size, num_workers=num_workers)
    # Neural networks
    if model.model_name in pytorch_models:
        if not cuda_test():
            warnings.warn("CUDA or MPS is not available. Testing on CPU could be very slow.", stacklevel=1)
        trainer = l.Trainer(precision=model.precision, enable_checkpointing=False)
        trainer.test(model, dm)
    # Sklearn models
    elif model.model_name in sklearn_models:
        dm.setup(stage="test")
        # Loading test datasets
        x_test, y_test, _ = sklearn_load_dataset(dm, "test", model.generate_features)
        model.test(x_test, y_test)
    else:
        raise ValueError("Wrong model name. Check spelling or read a documentation and choose a supported model")
