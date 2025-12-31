"""General segmentation functions and classes."""

from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, PositiveInt, TypeAdapter
from typing import Any, Literal, Optional, Union

import warnings
from pathlib import Path

from skimage import feature

import numpy as np

import lightning as l
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torchvision import tv_tensors
from torchvision.transforms import v2

from remote_sensing_processor.common.common_functions import read_json
from remote_sensing_processor.common.types import SKLModel, TorchNNModel, TorchTransform


class Epochs(BaseModel):
    """Basic class to configure epochs and early stopping."""

    max_epochs: NonNegativeInt
    early_stopping: bool
    min_delta: Optional[NonNegativeFloat] = None
    patience: Optional[PositiveInt] = None


def setup_trainer(model_file: Path, epochs: Optional[dict], val: bool, precision: Optional[str]) -> l.Trainer:
    """Set up a Lightning trainer."""
    monitor = "val_loss" if val else "train_loss"

    # Set up default epochs value if it is None
    if epochs is None:
        epochs = {"max_epochs": 5, "early_stopping": True}
    epochs = TypeAdapter(Epochs).validate_python(epochs)

    # Setting up trainer
    callbacks = []
    callbacks.append(
        l.pytorch.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor=monitor,
            mode="min",
            dirpath=model_file.parent,
            filename=model_file.stem,
            enable_version_counter=False,
        ),
    )

    if epochs.early_stopping:
        early_stopping_params = {}
        if epochs.min_delta is not None:
            early_stopping_params["min_delta"] = epochs.min_delta
        if epochs.patience is not None:
            early_stopping_params["patience"] = epochs.patience
        callbacks.append(
            l.pytorch.callbacks.early_stopping.EarlyStopping(monitor=monitor, mode="min", **early_stopping_params),
        )

    tb_logger = TensorBoardLogger(save_dir=model_file.parent / "logs" / "tensorboard", name=model_file.stem)
    csv_logger = CSVLogger(save_dir=model_file.parent / "logs" / "csv", name=model_file.stem)

    trainer = l.Trainer(
        max_epochs=epochs.max_epochs,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        precision=precision,
    )

    if not val:
        trainer.limit_val_batches = 0  # Don't run validation loop during training
        trainer.num_sanity_val_steps = 0  # Don't run sanity check before training
    return trainer


class Dataset:
    """Basic dataset class."""

    def __init__(self, dataset: BaseModel) -> None:
        path = dataset.path
        self.reference = path / "ref.tif"

        # Loading metadata
        self.meta = read_json(path / "meta.json")

        # Getting the subdatasets we need
        self.files = []
        length = []
        for name in self.meta["samples"]:
            if dataset.sub == "all" or name in dataset.sub:
                self.files.append((path / name))
                length.append(len(self.meta["samples"][name]))
        self.meta["len"] = length

        # Setting up common parameters
        self.input_shape = self.meta["tile_size"]
        self.border = self.meta["border"]
        self.input_dims = self.meta["x"]["bands"]
        self.variables = self.meta["x"]["variables"]
        self.x_nodata = self.meta["x"]["nodata"]


class DataModule(l.LightningDataModule):
    """Basic RSP datamodule."""

    def __init__(
        self,
        train_datasets: Optional[list[BaseModel]] = None,
        val_datasets: Optional[list[BaseModel]] = None,
        test_datasets: Optional[list[BaseModel]] = None,
        pred_dataset: Optional[BaseModel] = None,
        repeat: Optional[int] = 1,
        augment: Optional[Union[bool, tuple[Union[str, TorchTransform]]]] = False,
        batch_size: Optional[int] = 32,
        num_workers: Optional[Union[int, Literal["auto"]]] = "auto",
        bbox: Optional[bool] = False,
    ) -> None:
        super().__init__()
        # Setting up datasets
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        self.pred_dataset = pred_dataset
        self.repeat = repeat
        self.batch_size = batch_size
        self.reference = None
        # Configuring multiprocessing
        if num_workers != 0 and num_workers != "auto" and num_workers > torch.multiprocessing.cpu_count():
            warnings.warn(
                "'num_workers' is "
                + str(num_workers)
                + ", but you have only "
                + str(torch.multiprocessing.cpu_count())
                + "CPU cores. Setting 'num_workers' to 'auto'",
                stacklevel=1,
            )
            num_workers = "auto"
        if num_workers == "auto":
            cpus = torch.multiprocessing.cpu_count()
            gpus = max(torch.cuda.device_count(), 1)
            self.workers = max(1, cpus // gpus - 1)
        elif num_workers != 0:
            self.workers = num_workers
        else:
            self.workers = 0
        # Parameters that are needed to build the model
        self.input_shape = None
        self.border = None
        self.input_dims = None
        self.variables = None
        self.x_nodata = None
        self.y_nodata = None
        self.y_dtype = None
        # Dataset check
        if self.train_datasets:
            self.train_datasets = self.dataset_check(self.train_datasets)
        if self.val_datasets:
            self.val_datasets = self.dataset_check(self.val_datasets)
        if self.test_datasets:
            self.test_datasets = self.dataset_check(self.test_datasets)
        if self.pred_dataset:
            self.pred_dataset = self.dataset_check([self.pred_dataset])
        # Setting up transform
        self.transform = self.setup_transform(augment, bbox)

    def setup_transform(
        self,
        augment: Optional[Union[bool, tuple[Union[str, TorchTransform]]]],
        bbox: Optional[bool],
    ) -> Optional[v2.Compose]:
        """Setup data augmentation."""
        valid_transforms = [
            "ScaleJitter",
            "RandomResizedCrop",
            "RandomIoUCrop",
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandomZoomOut",
            "RandomRotation",
            "RandomAffine",
            "RandomPerspective",
            "ElasticTransform",
            "GaussianBlur",
        ]
        if augment is True:
            augment = ("RandomResizedCrop", "RandomHorizontalFlip")
        transforms = []
        if augment is not False:
            for t in augment:
                if isinstance(t, str):
                    if t not in valid_transforms:
                        raise ValueError(t + " is not a valid transform")
                    if t == "ScaleJitter":
                        transforms.append(v2.ScaleJitter(target_size=(self.input_shape, self.input_shape)))
                    elif t == "RandomResizedCrop":
                        transforms.append(
                            v2.RandomResizedCrop(size=(self.input_shape, self.input_shape), antialias=True),
                        )
                    elif t == "RandomIoUCrop":
                        transforms.append(v2.RandomIoUCrop())
                    elif t == "RandomHorizontalFlip":
                        transforms.append(v2.RandomHorizontalFlip(p=0.5))
                    elif t == "RandomVerticalFlip":
                        transforms.append(v2.RandomVerticalFlip(p=0.5))
                    elif t == "RandomZoomOut":
                        transforms.append(
                            v2.RandomZoomOut(fill={tv_tensors.Image: self.x_nodata, tv_tensors.Mask: self.y_nodata}),
                        )
                    elif t == "RandomRotation":
                        transforms.append(
                            v2.RandomRotation(
                                90,
                                fill={tv_tensors.Image: self.x_nodata, tv_tensors.Mask: self.y_nodata},
                            ),
                        )
                    elif t == "RandomAffine":
                        transforms.append(
                            v2.RandomAffine(
                                degrees=90,
                                translate=(0.5, 0.5),
                                shear=0.5,
                                fill={tv_tensors.Image: self.x_nodata, tv_tensors.Mask: self.y_nodata},
                            ),
                        )
                    elif t == "RandomPerspective":
                        transforms.append(
                            v2.RandomPerspective(
                                fill={tv_tensors.Image: self.x_nodata, tv_tensors.Mask: self.y_nodata},
                            ),
                        )
                    elif t == "ElasticTransform":
                        transforms.append(
                            v2.ElasticTransform(fill={tv_tensors.Image: self.x_nodata, tv_tensors.Mask: self.y_nodata}),
                        )
                    elif t == "GaussianBlur":
                        transforms.append(v2.GaussianBlur(kernel_size=(5, 9)))
                else:
                    transforms.append(t)

            # Fixing if size has changed or bboxes became invalid after transforms
            transforms.append(v2.Resize(size=(self.input_shape, self.input_shape), antialias=True))
            if bbox:
                transforms.append(v2.SanitizeBoundingBoxes())
            transforms.append(v2.ToPureTensor())

            return v2.Compose(transforms)
        return None

    def dataset_check(self, datasets: Any) -> Any:
        """Basic dataset check function. Should be extended in child classes."""
        return datasets

    def assert_common(self, ds: list[Dataset]) -> None:
        """Assert if common values are the same in every dataset."""
        # Checking data values
        if not all(d.input_shape == ds[0].input_shape for d in ds):
            raise ValueError("input shapes of input datasets are different")
        self.input_shape = ds[0].input_shape

        if not all(d.border == ds[0].border for d in ds):
            raise ValueError("borders of input datasets are different")
        self.border = ds[0].border

        if not all(d.input_dims == ds[0].input_dims for d in ds):
            raise ValueError("input dims of input datasets are different")
        self.input_dims = ds[0].input_dims

        if not all(d.variables == ds[0].variables for d in ds):
            raise ValueError("input dims of input datasets are different")
        self.variables = ds[0].variables

        if not all(d.x_nodata == ds[0].x_nodata for d in ds):
            raise ValueError("x nodatas of input datasets are different")
        self.x_nodata = ds[0].x_nodata

    def setup(self, stage: str) -> None:
        """Setup function."""
        if stage == "fit":
            self.ds_train = self.setup_datasets(self.train_datasets)
            if self.val_datasets is not None:
                self.ds_val = self.setup_datasets(self.val_datasets)
            else:
                self.ds_val = None
        if stage == "test":
            self.ds_test = self.setup_datasets(self.test_datasets)
        if stage == "predict":
            self.ds_pred = self.setup_datasets(self.pred_dataset)
            if "y" in self.ds_pred.column_names:
                self.ds_pred = self.ds_pred.remove_columns("y")

    def setup_datasets(self, ds: Any) -> Any:
        """Basic setup dataset function. Should be extended in child classes."""
        return ds

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Train dataloader."""
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
        )

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        """Validation dataloader."""
        if self.ds_val is not None:
            return torch.utils.data.DataLoader(
                self.ds_val,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.workers,
                persistent_workers=self.workers > 0,
            )
        return None

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Test dataloader."""
        return torch.utils.data.DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Prediction dataloader."""
        return torch.utils.data.DataLoader(
            self.ds_pred,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
        )


class Model(l.LightningModule):
    """Basic RSP Torch-based model class."""

    def __init__(
        self,
        model: Union[str, TorchNNModel],
        input_shape: int,
        input_dims: int,
        y_nodata: Optional[Union[int, float]],
        lr: Optional[float],
        precision: Optional[str],
        scheduler_opt: Literal["val_loss", "train_loss"],
        overwrite_loss: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if isinstance(model, str):
            self.model_name = model
        else:
            self.model_name = "Custom_Torch"
        self.input_shape = input_shape
        self.input_dims = input_dims
        self.y_nodata = y_nodata
        self.lr = lr
        self.precision = precision
        self.scheduler_opt = scheduler_opt
        # If loss is not set, use default loss in Transformers, if set then use the user-defined loss
        self.overwrite_loss = overwrite_loss
        # torch.autograd.set_detect_anomaly(True)

    def forward(self, batch: Any) -> Any:
        """Basic forward function of a model. Should be extended in child classes."""
        return batch, batch, 0, 0

    def training_step(self, batch: dict, batch_idx: Optional[int]) -> torch.Tensor:
        """Training step."""
        y, pred, loss, _ = self.forward(batch)
        if loss is None:
            raise ValueError("unable to compute loss")
        self.log_all(y, pred, loss, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: Optional[int]) -> None:
        """Validation step."""
        y, pred, loss, _ = self.forward(batch)
        if loss is None:
            raise ValueError("unable to compute loss")
        self.log_all(y, pred, loss, "val")

    def test_step(self, batch: dict, batch_idx: Optional[int]) -> None:
        """Test step."""
        y, pred, loss, _ = self.forward(batch)
        if loss is None:
            raise ValueError("unable to compute loss")
        self.log_all(y, pred, loss, "test")

    def predict_step(
        self,
        batch: dict,
        batch_idx: Optional[int],
        dataloader_idx: Optional[int] = 0,
    ) -> tuple[torch.Tensor, int]:
        """Prediction step."""
        _, pred, _, key = self.forward(batch)
        pred = self.post_process_predict(pred)
        return pred, key

    def configure_optimizers(self) -> dict:
        """Configuring optimizers and LR schedulers."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=5,
            ),
            "monitor": self.scheduler_opt,
            "interval": "epoch",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Ignore mismatches at the checkpoint loading stage.

        Copied from https://github.com/Lightning-AI/pytorch-lightning/issues/4690#issuecomment-731152036
        """
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    warnings.warn(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}",
                        stacklevel=2,
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                warnings.warn(f"Dropping parameter {k}", stacklevel=2)
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)


class SklearnModel:
    """Basic RSP Sklearn-based model class."""

    def __init__(self, model: Union[str, SKLModel], generate_features: Optional[bool], y_nodata: Optional[int]) -> None:
        if isinstance(model, str):
            self.model_name = model
        else:
            self.model_name = "Custom_Sklearn"
        self.generate_features = generate_features
        self.y_nodata = y_nodata

    def fit(self, x: Any, y: Any) -> None:
        """Basic fit class. Should be extended in child classes."""
        pass

    def test(self, x: Any, y: Any) -> None:
        """Basic test class. Should be extended in child classes."""
        pass

    def predict(self, x: Any) -> Any:
        """Basic prediction class. Should be extended in child classes."""
        return x


def sklearn_load_dataset(
    dm: DataModule,
    stage: str,
    generate_features: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Convert HF dataset to numpy arrays for Sklearn models."""
    y_nodata = dm.y_nodata

    def sklearn_process_tile(tile: dict) -> dict:
        x = tile["x"]
        y = tile.get("y")
        if generate_features:
            x = feature.multiscale_basic_features(
                x,
                intensity=True,
                edges=True,
                texture=True,
                channel_axis=0,
            )
        else:
            x = np.moveaxis(x, 0, -1)
        x = x.reshape((-1, x.shape[-1]))
        if y is not None:
            y = y.squeeze().ravel()
            if y_nodata is not None:
                mask = y != y_nodata
                x = x[mask]
                y = y[mask]
                # Indexes should start from 0
                y = y - 1
        out = {"key": tile["key"].item(), "x": x}
        if y is not None:
            out["y"] = y
        return out

    if stage == "train":
        dm.ds_train.set_format("numpy")
        dm.ds_train = dm.ds_train.map(sklearn_process_tile, keep_in_memory=True)
        x = np.concatenate(dm.ds_train["x"], axis=0)
        y = np.concatenate(dm.ds_train["y"], axis=0) if "y" in dm.ds_train.features else None
        keys = list(dm.ds_train["key"])
    elif stage == "val":
        if dm.ds_val is not None:
            dm.ds_val.set_format("numpy")
            dm.ds_val = dm.ds_val.map(sklearn_process_tile, keep_in_memory=True)
            x = np.concatenate(dm.ds_val["x"], axis=0)
            y = np.concatenate(dm.ds_val["y"], axis=0) if "y" in dm.ds_val.features else None
            keys = list(dm.ds_val["key"])
        else:
            raise ValueError("trying to perform validation while val dataset is not set")
    elif stage == "test":
        dm.ds_test.set_format("numpy")
        dm.ds_test = dm.ds_test.map(sklearn_process_tile, keep_in_memory=True)
        x = np.concatenate(dm.ds_test["x"], axis=0)
        y = np.concatenate(dm.ds_test["y"], axis=0) if "y" in dm.ds_test.features else None
        keys = list(dm.ds_test["key"])
    elif stage == "predict":
        dm.ds_pred.set_format("numpy")
        dm.ds_pred = dm.ds_pred.map(sklearn_process_tile, keep_in_memory=True)
        x = np.concatenate(dm.ds_pred["x"], axis=0)
        y = np.concatenate(dm.ds_pred["y"], axis=0) if "y" in dm.ds_pred.features else None
        keys = list(dm.ds_pred["key"])
    else:
        raise ValueError("Invalid stage")
    return x, y, keys
