"""Load bands and constants for index calculation."""

from typing import Optional, Union

import warnings

import spyndex

from pystac import Item


s2 = {
    "A": ["coastal", "B01", "B1", "B01.tif", "B1.tif", "B01.jp2", "B1.jp2"],
    "B": ["blue", "B02", "B2", "B02.tif", "B2.tif", "B02.jp2", "B2.jp2"],
    "G": ["green", "B03", "B3", "B03.tif", "B3.tif", "B03.jp2", "B3.jp2"],
    "R": ["red", "B04", "B4", "B04.tif", "B4.tif", "B04.jp2", "B4.jp2"],
    "RE1": ["rededge071", "B05", "B5", "B05.tif", "B5.tif", "B05.jp2", "B5.jp2"],
    "RE2": ["rededge075", "B06", "B6", "B06.tif", "B6.tif", "B06.jp2", "B6.jp2"],
    "RE3": ["rededge078", "B07", "B7", "B07.tif", "B7.tif", "B07.jp2", "B7.jp2"],
    "N": ["nir", "B08", "B8", "B08.tif", "B8.tif", "B08.jp2", "B8.jp2"],
    "N2": ["nir08", "B8A", "B8A.tif", "B8A.jp2"],
    "WV": ["nir09", "B09", "B9", "B09.tif", "B9.tif", "B09.jp2", "B9.jp2"],
    "S1": ["swir16", "B11", "B11.tif", "B11.jp2"],
    "S2": ["swir22", "B12", "B12.tif", "B12.jp2"],
}

l8 = {
    "A": ["coastal", "B01", "B1", "B01.tif", "B1.tif", "B01.TIF", "B1.TIF"],
    "B": ["blue", "B02", "B2", "B02.tif", "B2.tif", "B02.TIF", "B2.TIF"],
    "G": ["green", "B03", "B3", "B03.tif", "B3.tif", "B03.TIF", "B3.TIF"],
    "R": ["red", "B04", "B4", "B04.tif", "B4.tif", "B04.TIF", "B4.TIF"],
    "N": ["nir08", "B05", "B5", "B05.tif", "B5.tif", "B05.TIF", "B5.TIF"],
    "S1": ["swir16", "B06", "B6", "B06.tif", "B6.tif", "B06.TIF", "B6.TIF"],
    "S2": ["swir22", "B07", "B7", "B07.tif", "B7.tif", "B07.TIF", "B7.TIF"],
    "T1": ["lwir11", "B10", "B10.tif", "B10.TIF"],
    "T2": ["lwir12", "B11", "B11.tif", "B11.TIF"],
}

l57 = {
    "B": ["blue", "B01", "B1", "B01.tif", "B1.tif", "B01.TIF", "B1.TIF"],
    "G": ["green", "B02", "B2", "B02.tif", "B2.tif", "B02.TIF", "B2.TIF"],
    "R": ["red", "B03", "B3", "B03.tif", "B3.tif", "B03.TIF", "B3.TIF"],
    "N": ["nir08", "B04", "B4", "B04.tif", "B4.tif", "B04.TIF", "B4.TIF"],
    "S1": ["swir16", "B05", "B5", "B05.tif", "B5.tif", "B05.TIF", "B5.TIF"],
    "S2": ["swir22", "B07", "B7", "B07.tif", "B7.tif", "B07.TIF", "B7.TIF"],
    "T1": ["lwir11", "B06", "B6", "B06.tif", "B6.tif", "B06.TIF", "B6.TIF"],
}

l1 = {
    "G": [
        "green",
        "B01",
        "B1",
        "B01.tif",
        "B1.tif",
        "B01.TIF",
        "B1.TIF",
        "B04",
        "B4",
        "B04.tif",
        "B4.tif",
        "B04.TIF",
        "B4.TIF",
    ],
    "R": [
        "red",
        "B02",
        "B2",
        "B02.tif",
        "B2.tif",
        "B02.TIF",
        "B2.TIF",
        "B05",
        "B5",
        "B05.tif",
        "B5.tif",
        "B05.TIF",
        "B5.TIF",
    ],
    "N": [
        "nir08",
        "B03",
        "B3",
        "B03.tif",
        "B3.tif",
        "B03.TIF",
        "B3.TIF",
        "B06",
        "B6",
        "B06.tif",
        "B6.tif",
        "B06.TIF",
        "B6.TIF",
    ],
    "N2": [
        "nir09",
        "B04",
        "B4",
        "B04.tif",
        "B4.tif",
        "B04.TIF",
        "B4.TIF",
        "B07",
        "B7",
        "B07.tif",
        "B7.tif",
        "B07.TIF",
        "B7.TIF",
    ],
}
defd = {
    "A": ["coastal"],
    "B": ["blue"],
    "G": ["green"],
    "R": ["red"],
    "RE1": ["rededge071"],
    "RE2": ["rededge075"],
    "RE3": ["rededge078"],
    "N": ["nir", "nir08", "nir09"],
    "S1": ["swir16"],
    "S2": ["swir22"],
    "T1": ["lwir11", "lwir"],
    "T2": ["lwir12"],
}


def get_index(
    dataset: Item,
    index: str,
    bands: Optional[dict[str, Union[str, int, float]]] = None,
) -> dict[str, Union[str, int, float]]:
    """Get bands for a specific vegetation index."""
    t = dataset.common_metadata.description

    if bands is not None:
        bands = fix_bands(bands, dataset)

    if index in spyndex.indices:
        if "Sentinel2" in t:
            bands = get_bands_from_names(bands, s2, dataset)
        elif "Landsat8" in t:
            bands = get_bands_from_names(bands, l8, dataset)
        elif "Landsat7" in t or "Landsat5" in t:
            bands = get_bands_from_names(bands, l57, dataset)
        elif "Landsat1" in t:
            bands = get_bands_from_names(bands, l1, dataset)
        else:
            bands = get_bands_from_names(bands, defd, dataset)

        if index == "ARVI":
            bands = add_constants(bands, {"gamma": 1.0})
        elif index == "ATSAVI":
            bands = add_constants(bands, {"sla": 1.22, "slb": 0.03})
        elif index == "BWDRVI":
            bands = add_constants(bands, {"alpha": 0.1})
        elif index == "CRSWIR":
            bands = add_constants(bands, {"lambdaN2": 865, "lambdaS1": 1610, "lambdaS2": 2190})
        elif index == "DVIplus":
            bands = add_constants(bands, {"lambdaN": 850, "lambdaR": 678, "lambdaG": 560})
        elif index == "EVI":
            bands = add_constants(bands, {"g": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0})
        elif index == "EVI2":
            bands = add_constants(bands, {"g": 2.4, "L": 1.0})
        elif index == "FAI" or index == "FDI":
            bands = add_constants(bands, {"lambdaN": 850, "lambdaR": 678, "lambdaS1": 1610})
        elif index == "GDVI":
            bands = add_constants(bands, {"nexp": 1.0})
        if index == "IAVI":
            bands = add_constants(bands, {"gamma": 1.0})
        elif index == "IBI":
            bands = add_constants(bands, {"L": 0.5})
        elif index == "MBWI":
            bands = add_constants(bands, {"omega": 3.0})
        elif index == "MNLI":  # noqa SIM114
            bands = add_constants(bands, {"L": 0.5})
        elif index == "NBUI":
            bands = add_constants(bands, {"L": 0.5})
        elif index == "NDGI":
            bands = add_constants(bands, {"lambdaN": 850, "lambdaR": 678, "lambdaG": 560})
        elif index == "NDPI":
            bands = add_constants(bands, {"alpha": 0.74})
        elif index == "NDSInw":
            bands = add_constants(bands, {"beta": 0.05})
        elif index == "NDTI4RE":
            bands = add_constants(bands, {"gamma": 0.4})
        elif index == "NDVI4RE":
            bands = add_constants(bands, {"alpha": 0.7, "beta": 0.7})
        elif index == "NDWIns":
            bands = add_constants(bands, {"alpha": 2.0})
        elif index == "NIRvH2":
            bands = add_constants(bands, {"k": 0.25, "lambdaN": 850, "lambdaR": 678})
        elif index == "NIRvP":
            bands = add_constants(bands, {"PAR": 1.0})
        elif index == "OCVI":
            bands = add_constants(bands, {"cexp": 1.0})
        elif index == "RVI4RE":
            bands = add_constants(bands, {"alpha": 0.9, "beta": 0.6})
        elif index == "SARVI":  # noqa SIM114
            bands = add_constants(bands, {"L": 0.5})
        elif index == "SAVI":
            bands = add_constants(bands, {"L": 0.5})
        elif index == "SAVI2":
            bands = add_constants(bands, {"sla": 1.22, "slb": 0.03})
        elif index == "SAVI4RE":
            bands = add_constants(bands, {"alpha": 0.7, "beta": 0.7})
        elif index == "SAVIT":
            bands = add_constants(bands, {"L": 0.5})
        elif index == "SNDTI":
            bands = add_constants(bands, {"L": 0.6})
        elif index == "SNDTI4RE" or index == "STI4RE":
            bands = add_constants(bands, {"gamma": 0.4})
        elif index == "TSAVI":
            bands = add_constants(bands, {"sla": 1.22, "slb": 0.03})
        elif index == "WDRVI":
            bands = add_constants(bands, {"alpha": 0.1})
        elif index == "WDVI":
            bands = add_constants(bands, {"sla": 1.22})
        elif index == "kEVI":
            bands = add_constants(bands, {"g": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0})
        elif index == "sNIRvNDPI":
            bands = add_constants(bands, {"alpha": 0.74})
        if index not in ["kEVI", "kIPVI", "kNDVI", "kRVI", "kVARI"]:
            bands = filter_bands(bands, spyndex.indices[index].bands)
        elif index == "kEVI":
            bands["kNN"] = 1.0
            bands["kNR"] = 1.0
            bands["kNB"] = 1.0
            bands["kNL"] = 1.0
            bands = filter_bands(bands, [*spyndex.indices[index].bands, "N", "R", "B", "L"])
        elif index == "kIPVI" or index == "kNDVI" or index == "kRVI":
            bands["kNN"] = 1.0
            bands["kNR"] = 1.0
            bands = filter_bands(bands, [*spyndex.indices[index].bands, "N", "R"])
        elif index == "kVARI":
            bands["kGG"] = 1.0
            bands["kGR"] = 1.0
            bands["kGB"] = 1.0
            bands = filter_bands(bands, [*spyndex.indices[index].bands, "G", "R", "B"])
    else:
        # Calculating normalized difference index
        warnings.warn(
            "Index " + index + " is not currently supported. Calculating custom normalized difference index.",
            stacklevel=2,
        )
        if bands is None or len(bands) != 2:
            raise ValueError("Cannot calculate a custom index")
        # Setting bands as NDVI bands
        bands["N"] = bands[next(iter(bands.keys()))]
        bands["R"] = bands[list(bands.keys())[1]]
        bands = filter_bands(bands, spyndex.indices["NDVI"].bands)
    return bands


def fix_bands(bands: dict[str, Union[str, int, float]], dataset: Item) -> dict[str, Union[str, int, float]]:
    """Fixing band names if a band number or file name is passed instead of a band name."""
    for k, v in bands.items():
        if isinstance(v, str) and v not in dataset.assets:
            for asset in dataset.assets:
                if dataset.assets[asset].ext.eo.bands[0].name == v or dataset.assets[asset].href == v:
                    bands[k] = asset
                    break
            if bands[k] not in dataset.assets:
                raise ValueError(v + " is not in a dataset")
    return bands


def get_bands_from_names(
    bands: Union[dict[str, Union[str, int, float]], None],
    band_names: dict[str, list[str]],
    dataset: Item,
) -> dict[str, Union[str, int, float]]:
    """Get band names from names."""
    if bands is None:
        bands = {}
    for k, v in band_names.items():
        if k not in bands:
            for asset in dataset.assets:
                if asset in v or dataset.assets[asset].ext.eo.bands[0].name in v or dataset.assets[asset].href in v:
                    bands[k] = asset
                    break
    return bands


def add_constants(
    bands: dict[str, Union[str, int, float]],
    constants: dict[str, Union[int, float]],
) -> dict[str, Union[str, int, float]]:
    """Add constants if they are not set."""
    for k, v in constants.items():
        if k not in bands:
            warnings.warn(str(k) + " is not set. Falling back to default value " + str(v), stacklevel=2)
            bands[k] = v
    return bands


def filter_bands(bands: dict[str, Union[str, int, float]], f: list[str]) -> dict[str, Union[str, int, float]]:
    """Remove bands that are not needed."""
    unwanted = set(bands) - set(f)
    for unwanted_key in unwanted:
        del bands[unwanted_key]
    if set(bands) != set(f):
        raise ValueError("Looks like not all required bands are present in dataset")
    return bands
