"""RSP dataset type names."""

from typing import Union

import re
import tarfile
import zipfile
from pathlib import Path

from pystac import Item

from remote_sensing_processor.common.dataset import read_json


rsp_landsat_p_types = [
    "Landsat_olitirs_p",
    "Landsat_etm_p",
    "Landsat_tm_p",
    "Landsat_mss_p",
]

rsp_landsat_up_types = [
    "Landsat_olitirs_up_l1",
    "Landsat_etm_up_l1",
    "Landsat_tm_up_l1",
    "Landsat_mss_up_l1",
    "Landsat_olitirs_up_l2",
    "Landsat_etm_up_l2",
    "Landsat_tm_up_l2",
    "Landsat_mss_up_l2",
]

rsp_landsat_undefined_types = [
    "Undefined_Landsat_olitirs",
    "Undefined_Landsat_etm",
    "Undefined_Landsat_tm",
    "Undefined_Landsat_mss",
]

rsp_landsat_types = rsp_landsat_p_types + rsp_landsat_up_types

rsp_sentinel2_p_types = ["Sentinel2_p"]

rsp_sentinel2_up_types = [
    "Sentinel2_up_l2",
    "Sentinel2_up_l1",
]

rsp_sentinel2_undefined_types = ["Undefined_Sentinel2"]

rsp_sentinel2_types = rsp_sentinel2_p_types + rsp_sentinel2_up_types

rsp_undefined = ["Undefined"]

rsp_all_types = (
    rsp_landsat_types
    + rsp_landsat_undefined_types
    + rsp_sentinel2_types
    + rsp_sentinel2_undefined_types
    + rsp_undefined
)


def get_type(path: Union[Item, Path, str]) -> str:
    """Get an RSP dataset type."""
    if isinstance(path, Item):
        bands = [path.assets[i].href for i in path.assets]
        if len(bands) == 0:
            bands = None
        if path.common_metadata.description in rsp_all_types:
            return path.common_metadata.description
    elif isinstance(path, Path):
        # Reading bands
        if path.is_dir():
            bands = list(path.glob("**/*.*[!(zip|tar|tar.gz|aux.xml)*]"))
        elif ".json" in path.suffixes:
            bands = None
            dataset = read_json(path)
            # noinspection PyUnresolvedReferences
            if dataset is not None and path.common_metadata.description in rsp_all_types:
                return dataset.common_metadata.description
        elif (".tar" in path.suffixes) or (".gz" in path.suffixes):
            with tarfile.open(path) as file:
                bands = []
                for i in file.getmembers():
                    bands.append(path / i.name)
        elif ".zip" in path.suffixes:
            with zipfile.ZipFile(path) as file:
                bands_zip = file.namelist()
                bands = []
                for i in bands_zip:
                    bands.append(path / i)
        else:
            bands = None
    else:
        bands = None
    # Getting only band names from bands
    if bands is not None:
        for i in range(len(bands)):
            if isinstance(bands[i], Path):
                # noinspection PyUnresolvedReferences
                if ".json" in bands[i].suffixes:
                    dataset = read_json(path)
                    if dataset is not None and path.common_metadata.description in rsp_all_types:
                        return dataset.common_metadata.description
                # noinspection PyUnresolvedReferences
                bands[i] = bands[i].stem.split(".")[0]

    # Sentinel 2
    if bands is not None and (
        set(filter(re.compile(r"B*\d").match, bands))
        == {"B1", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9"}
        or set(filter(re.compile(r"B*\d").match, bands))
        == {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"}
    ):
        return "Sentinel2_p"  # sentinel2 preprocessed in RSP
    if re.search(r"T\d\d\w\w\w_", str(path)) is not None or re.search(r"MTD_MSIL\d\w.xml", str(path)):
        if re.search("MSIL2A", str(path)) is not None:
            return "Sentinel2_up_l2"  # sentinel2 without processing level 2
        if re.search("MSIL1C", str(path)) is not None:
            return "Sentinel2_up_l1"  # sentinel2 without processing level 1
        return "Undefined_Sentinel2"

    # Landsat
    # Getting Landsat type from a path
    if re.search(r"L\w\d\d", str(path)):
        return get_landsat_type(str(path), bands)
    # If Landsat type is not in path, trying to get it from band names
    if bands is not None and any(re.search(r"L\w\d\d", str(band)) for band in bands):
        for band in bands:
            return get_landsat_type(str(band), bands)
    return "Undefined"


def get_landsat_type(name: str, bands: list) -> str:
    """Get Landsat generation."""
    if re.search(r"L\w\d\d", name):
        if re.search(r"L\w\d\d", name).group(0) in ["LM05", "LM04", "LM03", "LM02", "LM01"]:
            if bands is not None and len(sorted(bands)[0]) <= 3:
                return "Landsat_mss_p"  # landsat1 processed in RSP
            if re.search(r"L\d\w\w", name):
                if re.search(r"L\d\w\w", name).group(0) in ["L1TP", "L1GT", "L1GS"]:
                    return "Landsat_mss_up_l1"  # landsat1 without processing level 1
                if re.search(r"L\d\w\w", name).group(0) in ["L2SP", "L2SR"]:
                    return "Landsat_mss_up_l2"  # landsat1 without processing level 2
            return "Undefined_Landsat_mss"
        if re.search(r"L\w\d\d", name).group(0) in ["LT05", "LT04"]:
            if bands is not None and len(sorted(bands)[0]) <= 3:
                return "Landsat_tm_p"  # landsat5 processed in RSP
            if re.search(r"L\d\w\w", name):
                if re.search(r"L\d\w\w", name).group(0) in ["L1TP", "L1GT", "L1GS"]:
                    return "Landsat_tm_up_l1"  # landsat5 without processing level 1
                if re.search(r"L\d\w\w", name).group(0) in ["L2SP", "L2SR"]:
                    return "Landsat_tm_up_l2"  # landsat5 without processing level 2
            return "Undefined_Landsat_tm"
        if re.search(r"L\w\d\d", name).group(0) == "LE07":
            if bands is not None and len(sorted(bands)[0]) <= 3:
                return "Landsat_etm_p"  # landsat7 processed in RSP
            if re.search(r"L\d\w\w", name):
                if re.search(r"L\d\w\w", name).group(0) in ["L1TP", "L1GT", "L1GS"]:
                    return "Landsat_etm_up_l1"  # landsat7 without processing level 1
                if re.search(r"L\d\w\w", name).group(0) in ["L2SP", "L2SR"]:
                    return "Landsat_etm_up_l2"  # landsat7 without processing level 2
            return "Undefined_Landsat_etm"
        if re.search(r"L\w\d\d", name).group(0) in ["LC08", "LC09"]:
            if bands is not None and len(sorted(bands)[0]) <= 3:
                return "Landsat_olitirs_p"  # landsat8 processed in RSP
            if re.search(r"L\d\w\w", name):
                if re.search(r"L\d\w\w", name).group(0) in ["L1TP", "L1GT", "L1GS"]:
                    return "Landsat_olitirs_up_l1"  # landsat8 without processing level 1
                if re.search(r"L\d\w\w", name).group(0) in ["L2SP", "L2SR"]:
                    return "Landsat_olitirs_up_l2"  # landsat8 without processing level 2
            return "Undefined_Landsat_olitirs"
        return "Undefined"
    return "Undefined"
