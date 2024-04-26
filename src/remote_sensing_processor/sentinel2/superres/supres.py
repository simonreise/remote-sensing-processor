import os
import gc
import pathlib

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from remote_sensing_processor.common.common_functions import persist

from remote_sensing_processor.sentinel2.superres.patches import get_test_patches, get_test_patches60, recompose_images

# This code is adapted from this repository
# https://github.com/lanha/DSen2 and is distributed under the same
# license.

SCALE = 2000
#MDL_PATH = "./weights/"
MDL_PATH  = pathlib.Path(__file__).parents[0].joinpath('weights/') 

#L1C_MDL_PATH_20M_DSEN2 = MDL_PATH.joinpath("l1c_dsen2_20m_s2_038_lr_1e-04.hdf5")
#L1C_MDL_PATH_60M_DSEN2 = MDL_PATH.joinpath("l1c_dsen2_60m_s2_038_lr_1e-04.hdf5")
#L2A_MDL_PATH_20M_DSEN2 = MDL_PATH.joinpath("l2a_dsen2_20m_s2_038_lr_1e-04.hdf5")
#L2A_MDL_PATH_60M_DSEN2 = MDL_PATH.joinpath("l2a_dsen2_60m_s2_038_lr_1e-04.hdf5")

L1C_MDL_PATH_20M_DSEN2 = MDL_PATH.joinpath("L1C20M.pt")
L1C_MDL_PATH_60M_DSEN2 = MDL_PATH.joinpath("L1C60M.pt")
L2A_MDL_PATH_20M_DSEN2 = MDL_PATH.joinpath("L2A20M.pt")
L2A_MDL_PATH_60M_DSEN2 = MDL_PATH.joinpath("L2A60M.pt")

'''
if not os.path.exists(L1C_MDL_PATH_20M_DSEN2):
    urllib.request.urlretrieve('https://github.com/up42/s2-superresolution/raw/master/weights/l1c_dsen2_20m_s2_038_lr_1e-04.hdf5', L1C_MDL_PATH_20M_DSEN2)
if not os.path.exists(L1C_MDL_PATH_60M_DSEN2):
    urllib.request.urlretrieve('https://github.com/up42/s2-superresolution/raw/master/weights/l1c_dsen2_60m_s2_038_lr_1e-04.hdf5', L1C_MDL_PATH_60M_DSEN2)
if not os.path.exists(L2A_MDL_PATH_20M_DSEN2):
    urllib.request.urlretrieve('https://github.com/up42/s2-superresolution/raw/master/weights/l2a_dsen2_20m_s2_038_lr_1e-04.hdf5', L2A_MDL_PATH_20M_DSEN2)
if not os.path.exists(L2A_MDL_PATH_60M_DSEN2):
    urllib.request.urlretrieve('https://github.com/up42/s2-superresolution/raw/master/weights/l2a_dsen2_60m_s2_038_lr_1e-04.hdf5', L2A_MDL_PATH_60M_DSEN2)
'''    


def dsen2_20(d10, d20, image_level):
    # Input to the funcion must be of shape:
    #     d10: [x,y,4]      (B2, B3, B4, B8)
    #     d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    #     deep: specifies whether to use VDSen2 (True), or DSen2 (False)

    border = 8
    p10, p20 = get_test_patches(d10, d20, patch_size=128, border=border)
    p10 /= SCALE
    p20 /= SCALE
    p10, p20 = persist(p10, p20)
    test = [p10, p20]
    if image_level == "MSIL1C":
        model_filename = L1C_MDL_PATH_20M_DSEN2
    else:
        model_filename = L2A_MDL_PATH_20M_DSEN2
    input_shape = ((4, None, None), (6, None, None))
    prediction = _predict(test, model_filename, input_shape)
    del test, p10, p20
    images = recompose_images(prediction, border=border, ref=d10)
    images = persist(images)
    images *= SCALE
    images = persist(images)
    return images


def dsen2_60(d10, d20, d60, image_level):
    # Input to the funcion must be of shape:
    #     d10: [x,y,4]      (B2, B3, B4, B8)
    #     d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    #     d60: [x/6,y/6,2]  (B1, B9) -- NOT B10
    #     deep: specifies whether to use VDSen2 (True), or DSen2 (False)

    border = 12
    p10, p20, p60 = get_test_patches60(d10, d20, d60, patch_size=192, border=border)
    p10 /= SCALE
    p20 /= SCALE
    p60 /= SCALE
    p10, p20, p60 = persist(p10, p20, p60)

    test = [p10, p20, p60]
    if image_level == "MSIL1C":
        model_filename = L1C_MDL_PATH_60M_DSEN2
    else:
        model_filename = L2A_MDL_PATH_60M_DSEN2
    input_shape = ((4, None, None), (6, None, None), (2, None, None))
    prediction = _predict(test, model_filename, input_shape)
    del test, p10, p20, p60
    images = recompose_images(prediction, border=border, ref=d10)
    images = persist(images)
    images *= SCALE
    images = persist(images)
    return images


def _predict(test, model_filename, input_shape):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = torch.jit.load(model_filename, map_location = device)
    dataset = S2Dataset(test)
    # TODO: predict does not work faster with num_workers, and also have pickling errors
    num_workers = 0
    if num_workers == 'auto':
        cpus = torch.multiprocessing.cpu_count()
        gpus = max(torch.cuda.device_count(), 1)
        num_workers = max(1 , cpus // gpus - 1)
    loader = DataLoader(dataset, batch_size=32, num_workers=num_workers, pin_memory=True)
    predictions = []
    with torch.inference_mode():
        for data in loader:
            predictions.append(model(*[d.to(device) for d in data]).cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    del model
    del dataset
    del loader
    torch.cuda.empty_cache()
    gc.collect()
    return predictions

        
class S2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_len = dataset[0].shape[0]

    def __getitem__(self, index):
        chip = []
        for i in range(len(self.dataset)):
            chip.append(torch.Tensor(self.dataset[i][index].data.compute()))
        return chip

    def __len__(self):
        return self.dataset_len
