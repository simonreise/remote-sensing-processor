"""DSen2 superresolution."""

import gc
from pathlib import Path

import numpy as np
import xarray as xr

from pystac import Item

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset

from remote_sensing_processor.common.common_functions import persist
from remote_sensing_processor.common.fill import fillnodata
from remote_sensing_processor.imagery.sentinel2.patches import (
    get_test_patches,
    get_test_patches60,
    recompose_images,
)


# This code is adapted from this repository https://github.com/lanha/DSen2 and is distributed under the same license.

SCALE = 2000


def superresolution(data10: xr.Dataset, data20: xr.Dataset, data60: xr.Dataset, dataset: Item) -> xr.Dataset:
    """
    DSen2 Superresolution.

    This function takes the raster data at 10, 20, and 60 m resolutions, and by applying
    data_final method creates the input data for the convolutional neural network.
    It returns 10 m resolution for all the bands in 20 and 60 m resolutions.
    """
    rsp_type = dataset.common_metadata.description

    # Dsen2 cannot process B10, removing it
    if "cirrus" in data60.variables:
        data60 = data60.drop_vars(["cirrus"])

    # Super-resolving the 60m data into 10m bands
    sr60 = dsen2_60(data10, data20, data60, rsp_type)
    # Super-resolving the 20m data into 10m bands"
    sr20 = dsen2_20(data10, data20, rsp_type)

    sr_final = xr.merge([data10.astype("uint16"), sr20.astype("uint16"), sr60.astype("uint16")])

    sr_final = persist(sr_final)

    # Masking 65555 pixels
    sr_final = sr_final.where(sr_final <= 10000, 1)
    sr_final = fillnodata(sr_final, xr.where(sr_final == 1, 0, 1), 500, 0)

    sr_final = persist(sr_final.chunk("auto"))

    gc.collect()
    return sr_final


def get_model(rsp_type: str, res: int) -> Path:
    """Load model weights from Huggingface Hub."""
    model = None
    if rsp_type == "Sentinel2_up_l1":
        if res == 60:
            model = hf_hub_download(repo_id="simonreise/dsen2", filename="L1C60M.pt")
        elif res == 20:
            model = hf_hub_download(repo_id="simonreise/dsen2", filename="L1C20M.pt")
    elif rsp_type == "Sentinel2_up_l2":
        if res == 60:
            model = hf_hub_download(repo_id="simonreise/dsen2", filename="L2A60M.pt")
        elif res == 20:
            model = hf_hub_download(repo_id="simonreise/dsen2", filename="L2A20M.pt")
    if model is None:
        raise ValueError("Cannot load model")
    return Path(model)


def dsen2_20(d10: xr.Dataset, d20: xr.Dataset, rsp_type: str) -> xr.Dataset:
    """
    Process 20m data.

    Input to the function must be of shape:
    d10: [x,y,4]      (B2, B3, B4, B8)
    d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    deep: specifies whether to use VDSen2 (True), or DSen2 (False)
    """
    border = 8
    # Converting datasets to dataarrays
    d10 = d10.to_dataarray("band")
    d20 = d20.to_dataarray("band")
    # Getting band names
    b20 = d20.band.values
    # Getting test patches
    p10, p20 = get_test_patches(d10, d20, patch_size=128, border=border)
    p10 /= SCALE
    p20 /= SCALE
    test = xr.concat([p10, p20], dim="band").astype("float32")
    test = persist(test)
    # Predicting
    model_filename = get_model(rsp_type, 20)
    prediction = _predict(test, model_filename)
    p20.data = p20.data + prediction
    # Postprocessing images
    images = recompose_images(p20, border=border, ref=d20)
    images = persist(images)
    images *= SCALE
    # Adding band names
    images = images.assign_coords({"band": b20})
    # Converting back to dataset
    images = images.to_dataset("band")
    images = persist(images)
    del test, p10, p20, prediction
    return images


def dsen2_60(d10: xr.Dataset, d20: xr.Dataset, d60: xr.Dataset, rsp_type: str) -> xr.Dataset:
    """
    Process 60m data.

    Input to the function must be of shape:
    d10: [x,y,4]      (B2, B3, B4, B8)
    d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    d60: [x/6,y/6,2]  (B1, B9) -- NOT B10
    deep: specifies whether to use VDSen2 (True), or DSen2 (False)
    """
    border = 12
    # Converting datasets to dataarrays
    d10 = d10.to_dataarray("band")
    d20 = d20.to_dataarray("band")
    d60 = d60.to_dataarray("band")
    # Getting band names
    b60 = d60.band.values
    # Getting test patches
    p10, p20, p60 = get_test_patches60(d10, d20, d60, patch_size=192, border=border)
    p10 /= SCALE
    p20 /= SCALE
    p60 /= SCALE
    test = xr.concat([p10, p20, p60], dim="band").astype("float32")
    test = persist(test)
    # Predicting
    model_filename = get_model(rsp_type, 60)
    prediction = _predict(test, model_filename)
    p60.data = p60.data + prediction
    # Postprocessing images
    images = recompose_images(p60, border=border, ref=d60)
    images = persist(images)
    images *= SCALE
    # Adding band names
    images = images.assign_coords({"band": b60})
    # Converting back to dataset
    images = images.to_dataset("band")
    images = persist(images)
    del test, p10, p20, p60, prediction
    return images


def _predict(test: xr.DataArray, model_filename: Path) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.jit.load(model_filename, map_location=device)

    dataset = S2Dataset(test.compute())

    if device == "cuda":
        mem = 0
        for i in range(torch.cuda.device_count()):
            mem += round(torch.cuda.get_device_properties(i).total_memory / 1024**3)
        if mem >= 24:
            bs = 128
        elif mem >= 12:
            bs = 64
        elif mem >= 6:
            bs = 32
        elif mem >= 3:
            bs = 16
        else:
            bs = 8
    else:
        bs = 32

    loader = DataLoader(dataset, batch_size=bs, pin_memory=True)

    # TODO: convert model to onnxruntime and find out why does it sometimes run faster and sometimes slower than pytorch
    predictions = []
    with torch.inference_mode():
        for data in loader:
            predictions.append(model(data.to(device)).cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)

    del model
    del dataset
    del loader
    torch.cuda.empty_cache()
    gc.collect()
    return predictions


class S2Dataset(Dataset):
    """Custom Torch Dataset for Sentinel-2 data."""

    def __init__(self, dataset: xr.DataArray) -> None:
        self.dataset = dataset
        self.dataset_len = dataset.shape[0]

    def __getitem__(self, index: int) -> np.ndarray:  # noqa: D105
        return self.dataset[index].data

    def __len__(self) -> int:  # noqa: D105
        return self.dataset_len
