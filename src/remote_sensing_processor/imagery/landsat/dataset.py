"""Read Landsat STAC."""

from typing import Optional, Union

import re
import warnings
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import ElementTree as ET  # noqa: N817

import fsspec
from dateutil.parser import parse

from xarray import DataArray, Dataset

import rasterio as rio
import satpy
from rasterio.warp import transform_bounds

import pystac
from antimeridian import FixWindingWarning
from pystac.extensions.eo import EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterBand, RasterExtension
from pystac.extensions.view import ViewExtension
from stactools.core.utils import antimeridian

from remote_sensing_processor.common.dataset import (
    add_asset,
    add_rsp_metadata,
    filter_bands,
    read_json,
    validate,
)
from remote_sensing_processor.imagery.types import get_type, rsp_landsat_types


LANDSAT_BANDS = ["coastal", "blue", "green", "red", "nir08", "nir09", "swir16", "swir22", "cirrus", "lwir11", "lwir12"]
PAN_BAND = ["pan"]

ACCURACY_EXTENSION_SCHEMA = "https://stac-extensions.github.io/accuracy/v1.0.0-beta.1/schema.json"
LANDSAT_EXTENSION_SCHEMA = "https://stac-extensions.github.io/landsat/v2.0.0/schema.json"


def get_name(path: Union[pystac.Item, Path], mtl: ET) -> str:
    """Get Landsat name that will be used as an ID (e.g., LC09_L1TP_142020_20230825_20230825_02_T1)."""
    # Trying to read from STAC
    if isinstance(path, pystac.Item) and path.id is not None:
        return path.id
    if mtl is not None:
        # Trying to read from mtl
        return mtl.find(".//PRODUCT_CONTENTS/LANDSAT_PRODUCT_ID").text
    # Else trying to read from a path
    landsat_re = r"L\w\d\d_L\d\w\w_\d\d\d\d\d\d_\d\d\d\d\d\d\d\d_\d\d\d\d\d\d\d\d_\d\d_\w\d"
    p = re.search(landsat_re, path.stem.split(".")[0])
    if p:
        return p.group(0)
    return path.stem.split(".")[0]


def add_raster_info(dataset: pystac.Item, mtl: ET) -> None:
    """Add raster band to existing STAC."""
    for asset in dataset.assets:
        if dataset.properties["landsat:collection_number"] == "02":
            name = dataset.assets[asset].ext.eo.bands[0].name
            if (
                "Level-2" in dataset.assets[asset].description
                and "Surface Temperature" in dataset.assets[asset].description
            ):
                name = "ST_" + name
            else:
                name = name[1:]
            dtype = mtl.find(".//PRODUCT_CONTENTS/DATA_TYPE_BAND_" + name).text.lower()
        else:
            dtype = "uint16"
        nodata = 0
        raster = RasterExtension.ext(dataset.assets[asset], add_if_missing=True)
        # noinspection PyTypeChecker
        raster.bands = [RasterBand.create(nodata=nodata, data_type=dtype)]


def get_datetime(name: str, mtl: ET) -> datetime:
    """Generate datetime from MTL (more precise) or from name (less precise)."""
    if mtl is not None:
        date = mtl.find(".//IMAGE_ATTRIBUTES/DATE_ACQUIRED").text
        time = mtl.find(".//IMAGE_ATTRIBUTES/SCENE_CENTER_TIME").text
    else:
        if re.search(r"_\d\d\d\d\d\d\d\d_", name):
            date = re.search(r"_\d\d\d\d\d\d\d\d_", name).group(0).strip("_")
        else:
            # No date in the name
            date = "20000101"
        date = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        time = "00:00:00.0000000Z"
    return parse("%sT%s" % (date, time))


def get_bbox(mtl: ET, scene: satpy.Scene) -> list[Union[int, float]]:
    """Get bounding box from MTL or rasterio bbox."""
    if mtl is not None:
        lons = [
            float(mtl.find(".//PROJECTION_ATTRIBUTES/CORNER_UL_LON_PRODUCT").text),
            float(mtl.find(".//PROJECTION_ATTRIBUTES/CORNER_UR_LON_PRODUCT").text),
            float(mtl.find(".//PROJECTION_ATTRIBUTES/CORNER_LL_LON_PRODUCT").text),
            float(mtl.find(".//PROJECTION_ATTRIBUTES/CORNER_LR_LON_PRODUCT").text),
        ]

        lats = [
            float(mtl.find(".//PROJECTION_ATTRIBUTES/CORNER_UL_LAT_PRODUCT").text),
            float(mtl.find(".//PROJECTION_ATTRIBUTES/CORNER_UR_LAT_PRODUCT").text),
            float(mtl.find(".//PROJECTION_ATTRIBUTES/CORNER_LL_LAT_PRODUCT").text),
            float(mtl.find(".//PROJECTION_ATTRIBUTES/CORNER_LR_LAT_PRODUCT").text),
        ]
        return [min(lons), min(lats), max(lons), max(lats)]
    # noinspection PyTypeChecker
    return list(transform_bounds(scene["B4"].rio.crs, "EPSG:4326", *scene["B4"].rio.bounds()))


def get_coords(
    bbox: list[Union[int, float]],
    path: Union[pystac.Item, Path],
    ls_type: str,
) -> dict[str, list[list[list[float]]]]:
    """Get coordinates from ANG file or bbox."""
    sz = []
    coords = []
    if "landsat8" in ls_type:
        # Load ANG file
        if isinstance(path, pystac.Item) and "ang" in path.assets:
            with fsspec.open(path.assets["ang"].href) as f:
                ang = f.readlines()
                ang = [line.decode() for line in ang]
        elif path.is_file() and ((".tar" in path.suffixes) or (".gz" in path.suffixes)):
            fs = fsspec.filesystem("tar", fo=path.as_posix())
            ang_path = fs.glob("*_ANG.txt")
            if ang_path:
                ang_path = ang_path[0]
                ang = fs.open(ang_path)
                ang = [line.decode() for line in ang.readlines()]
            else:
                ang = None
        elif path.is_dir():
            ang_path = list(path.glob("*_ANG.txt"))
            if ang_path:
                ang_path = ang_path[0]
                ang = ang_path.open()
                ang = ang.readlines()
            else:
                ang = None
        else:
            ang = None

        for line in ang:
            if "BAND01_NUM_L1T_LINES" in line or "BAND01_NUM_L1T_SAMPS" in line:
                sz.append(float(line.split("=")[1]))
            if "BAND01_L1T_IMAGE_CORNER_LINES" in line or "BAND01_L1T_IMAGE_CORNER_SAMPS" in line:
                coords.append([float(line) for line in line.split("=")[1].strip().strip("()").split(",")])
            if len(coords) == 2:
                break
        dlon = bbox[2] - bbox[0]
        dlat = bbox[3] - bbox[1]
        lons = [c / sz[1] * dlon + bbox[0] for c in coords[1]]
        lats = [((sz[0] - c) / sz[0]) * dlat + bbox[1] for c in coords[0]]
        coordinates = [
            [
                [lons[0], lats[0]],
                [lons[1], lats[1]],
                [lons[2], lats[2]],
                [lons[3], lats[3]],
                [lons[0], lats[0]],
            ],
        ]
    else:
        # Just construct it from bbox
        coordinates = [
            [
                [bbox[0], bbox[1]],  # LL
                [bbox[2], bbox[1]],  # UL
                [bbox[2], bbox[3]],  # UR
                [bbox[0], bbox[3]],  # LR
                [bbox[0], bbox[1]],  # LL
            ],
        ]
    return {"type": "Polygon", "coordinates": coordinates}


def get_cloud_cover(mtl: ET) -> float:
    """Get cloud cover."""
    if mtl is not None:
        return float(mtl.find(".//IMAGE_ATTRIBUTES/CLOUD_COVER").text)
    return 0.0


def get_epsg(mtl: ET, scene: satpy.Scene) -> int:
    """Get CRS from MTL or scene."""
    if mtl is not None:
        try:
            utm_zone_integer = int(mtl.find(".//PROJECTION_ATTRIBUTES/UTM_ZONE").text.zfill(2))
            return int(f"326{utm_zone_integer}")
        except Exception:
            lat_ts = mtl.find(".//PROJECTION_ATTRIBUTES/TRUE_SCALE_LAT").text
            if lat_ts == "-71.00000":
                # Antarctic
                return 3031
            if lat_ts == "71.00000":
                # Arctic
                return 3995
            raise ValueError(f"Unexpeced value for PROJECTION_ATTRIBUTES/TRUE_SCALE_LAT: {lat_ts} ") from None
    else:
        # Read from one of the bands
        return scene["B4"].rio.crs.to_epsg()


def get_sr_shape(mtl: ET, scene: satpy.Scene, pan: bool = False) -> list[int]:
    """Getting image shape from MTL or scene."""
    if mtl is not None:
        if pan:
            return [
                int(mtl.find(".//PROJECTION_ATTRIBUTES/PANCHROMATIC_LINES").text),
                int(mtl.find(".//PROJECTION_ATTRIBUTES/PANCHROMATIC_SAMPLES").text),
            ]
        return [
            int(mtl.find(".//PROJECTION_ATTRIBUTES/REFLECTIVE_LINES").text),
            int(mtl.find(".//PROJECTION_ATTRIBUTES/REFLECTIVE_SAMPLES").text),
        ]
    return [scene["B4"].rio.height, scene["B4"].rio.width]


def get_transform(shape: list[int], scene: satpy.Scene, bbox: Optional[list[Union[int, float]]] = None) -> list[float]:
    """Getting transform from MTL or one of the bands."""
    if shape is not None and bbox is not None:
        return list(rio.transform.from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], shape[1], shape[0]))[:6]
    return list(scene["B4"].rio.transform())[:6]


def get_dtype(mtl: ET, col: str) -> str:
    """Get datatype."""
    if mtl is not None and col == "2":
        return mtl.find(".//PRODUCT_CONTENTS/DATA_TYPE_BAND_4").text.lower()
    return "uint16"


def read_landsat_dataset(path: Union[pystac.Item, Path], scene: satpy.Scene, rsp_type: str) -> pystac.Item:
    """Read Landsat STAC."""
    warnings.filterwarnings("ignore", category=FixWindingWarning)

    # Reading MTL
    mtl = scene._readers[next(iter(scene._readers))].file_handlers["l1_metadata"][0].root

    # If STAC file available, use it
    if isinstance(path, pystac.Item):
        stacs = [path]
    elif path.is_file() and ((".tar" in path.suffixes) or (".gz" in path.suffixes)):
        fs = fsspec.filesystem("tar", fo=path.as_posix())
        stacs = fs.glob("*stac.json")
        stacs = [read_json(fs.open(file)) for file in stacs]
    elif path.is_dir():
        stacs = list(path.glob("*stac.json"))
        stacs = [read_json(file) for file in stacs]
    else:
        stacs = None

    if stacs and 0 < len(stacs) <= 2:
        # L2 products have separate STACs for reflectance and thermal products
        if len(stacs) == 2:
            for aname, asset in stacs[1].assets.items():
                if aname not in stacs[0].assets:
                    stacs[0].assets[aname] = asset

        stac = stacs[0]

        # Some metadata is not relevant and will break the stac
        stac.clear_links()
        stac.collection_id = None
        # TODO: remove when v1.0.0 is released
        if "https://stac-extensions.github.io/accuracy/v1.0.0/schema.json" in stac.stac_extensions:
            stac.stac_extensions.remove("https://stac-extensions.github.io/accuracy/v1.0.0/schema.json")
            stac.stac_extensions.append(ACCURACY_EXTENSION_SCHEMA)
        # Replaces USGS landsat extension schema with newer STAC schema
        if "https://landsat.usgs.gov/stac/landsat-extension/v1.1.1/schema.json" in stac.stac_extensions:
            stac.stac_extensions.remove("https://landsat.usgs.gov/stac/landsat-extension/v1.1.1/schema.json")
            stac.stac_extensions.append(LANDSAT_EXTENSION_SCHEMA)
        # Breaks validation because need atm-emissivity and water-vapor links and also has an outdated EO link
        if "https://stac-extensions.github.io/card4l/v0.1.0/optical/schema.json" in stac.stac_extensions:
            stac.stac_extensions.remove("https://stac-extensions.github.io/card4l/v0.1.0/optical/schema.json")

        # Setting up type
        if rsp_type == "Undefined":
            rsp_type = get_type(stac.id)
        add_rsp_metadata(stac, rsp_type=rsp_type)

        # Removing _SR suffix from the name for l2
        if stac.id.endswith("_SR") or stac.id.endswith("_ST"):
            stac.id = stac.id[:-3]
        # Thermal bands have different names in Landsat 8, 7 and 5, so we have to rename them
        if "lwir" in stac.assets:
            stac.assets["lwir11"] = stac.assets.pop("lwir")
            stac.assets["lwir11"].ext.eo.bands[0].common_name = "lwir11"
        if "lwir_high" in stac.assets:
            stac.assets["lwir12"] = stac.assets.pop("lwir_high")
            stac.assets["lwir12"].ext.eo.bands[0].common_name = "lwir12"
        if "index" in stac.assets:
            stac.assets.pop("index")

        # Changes non-conventional names to conventional (e.g., OLI_B4 to B4)
        for asset in stac.assets:
            if "eo:bands" in stac.assets[asset].extra_fields:
                name = stac.assets[asset].ext.eo.bands[0].name
                if re.search(r"B\d{1,2}(?:_VCID_\d)?", name):
                    stac.assets[asset].ext.eo.bands[0].name = re.search(r"B\d{1,2}(?:_VCID_\d)?", name)[0]

        # Removing unneeded assets and add dtype and nodata info
        filter_bands(stac, LANDSAT_BANDS + PAN_BAND)
        add_raster_info(stac, mtl)
    else:
        # Reading landsat name
        name = get_name(path, mtl)

        # Getting datetime
        date_time = get_datetime(name, mtl)

        # Getting bounding box
        bbox = get_bbox(mtl, scene)
        # Getting coords
        coords = get_coords(bbox, path, rsp_type)

        # Creating a STAC item
        stac = pystac.Item(
            id=name,
            datetime=date_time,
            geometry=coords,
            bbox=bbox,
            properties={},
        )
        add_rsp_metadata(stac, rsp_type=rsp_type)

        # Adding EO extension
        eo_ext = EOExtension.ext(stac, add_if_missing=True)
        cloud_cover = get_cloud_cover(mtl)
        eo_ext.cloud_cover = cloud_cover

        # Adding projection info
        proj_ext = ProjectionExtension.ext(stac, add_if_missing=True)
        crs = get_epsg(mtl, scene)
        proj_ext.epsg = crs
        sr_shape = get_sr_shape(mtl, scene)
        proj_ext.shape = sr_shape
        proj_ext.transform = get_transform(sr_shape, scene, bbox)

        # Adding visual info
        if mtl is not None:
            view_ext = ViewExtension.ext(stac, add_if_missing=True)
            azimuth = float(mtl.find(".//IMAGE_ATTRIBUTES/SUN_AZIMUTH").text)
            if azimuth < 0.0:
                azimuth += 360
            view_ext.sun_azimuth = azimuth
            view_ext.sun_elevation = float(mtl.find(".//IMAGE_ATTRIBUTES/SUN_ELEVATION").text)
            if "Landsat_olitirs" in rsp_type:
                if mtl.find(".//IMAGE_ATTRIBUTES/NADIR_OFFNADIR").text == "OFFNADIR":
                    view_ext.off_nadir = abs(float(mtl.find(".//IMAGE_ATTRIBUTES/ROLL_ANGLE").text))
                else:
                    view_ext.off_nadir = 0

        # Adding Landsat and accuracy extensions
        stac.stac_extensions.append(LANDSAT_EXTENSION_SCHEMA)
        stac.stac_extensions.append(ACCURACY_EXTENSION_SCHEMA)
        d = {}
        # Writing common and Landsat metadata
        if mtl is not None:
            stac.common_metadata.platform = mtl.find(".//IMAGE_ATTRIBUTES/SPACECRAFT_ID").text
            stac.common_metadata.gsd = float(mtl.find(".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_REFLECTIVE").text)
            try:
                d["landsat:wrs_type"] = mtl.find(".//IMAGE_ATTRIBUTES/WRS_TYPE").text
            except Exception:
                if ("LM01" in name) or ("LM02" in name) or ("LM03" in name):
                    d["landsat:wrs_type"] = "1"
                else:
                    d["landsat:wrs_type"] = "2"
            try:
                d["landsat:wrs_path"] = mtl.find(".//IMAGE_ATTRIBUTES/WRS_PATH").text.zfill(3)
                d["landsat:wrs_row"] = mtl.find(".//IMAGE_ATTRIBUTES/WRS_ROW").text.zfill(3)
            except Exception:
                d["landsat:wrs_path"] = name.split("_")[2][0:3]
                d["landsat:wrs_row"] = name.split("_")[2][3:6]
            d["landsat:collection_category"] = mtl.find(".//PRODUCT_CONTENTS/COLLECTION_CATEGORY").text
            d["landsat:collection_number"] = mtl.find(".//PRODUCT_CONTENTS/COLLECTION_NUMBER").text
            d["landsat:correction"] = mtl.find(".//PRODUCT_CONTENTS/PROCESSING_LEVEL").text
            d["landsat:scene_id"] = mtl.find(".//LEVEL1_PROCESSING_RECORD/LANDSAT_SCENE_ID").text
            d["accuracy:geometric_rmse"] = float(mtl.find(".//LEVEL1_PROCESSING_RECORD/GEOMETRIC_RMSE_MODEL").text)
        else:
            if "Landsat_olitirs" in rsp_type:
                stac.common_metadata.platform = "LANDSAT_8"
                stac.common_metadata.gsd = 30
            elif "Landsat_etm" in rsp_type:
                stac.common_metadata.platform = "LANDSAT_7"
                stac.common_metadata.gsd = 30
            elif "Landsat_tm" in rsp_type:
                stac.common_metadata.platform = "LANDSAT_5"
                stac.common_metadata.gsd = 30
            elif "Landsat_mss" in rsp_type:
                stac.common_metadata.platform = "LANDSAT_1"
                stac.common_metadata.gsd = 60
            if ("LM01" in name) or ("LM02" in name) or ("LM03" in name):
                d["landsat:wrs_type"] = "1"
            else:
                d["landsat:wrs_type"] = "2"
            d["landsat:wrs_path"] = name.split("_")[2][0:3]
            d["landsat:wrs_row"] = name.split("_")[2][3:6]
            d["landsat:collection_category"] = name.split("_")[-1]
            d["landsat:collection_number"] = name.split("_")[-2]
            d["landsat:correction"] = name.split("_")[1]
            d["landsat:scene_id"] = name
            d["accuracy:geometric_rmse"] = 0.0
        d["landsat:cloud_cover_land"] = cloud_cover
        stac.properties.update(**d)
        if "Landsat_olitirs" in rsp_type:
            stac.common_metadata.instruments = ["oli", "tirs"]
        elif "Landsat_etm" in rsp_type:
            stac.common_metadata.instruments = ["etm+"]
        elif "Landsat_tm" in rsp_type:
            stac.common_metadata.instruments = ["tm"]
        elif "Landsat_mss" in rsp_type:
            stac.common_metadata.instruments = ["mss"]
        if "l2" in rsp_type:
            stac.extra_fields["description"] = "Landsat Collection 2 Level-2 Surface Reflectance Product"
        else:
            stac.extra_fields["description"] = "Landsat Collection 2 Level-1 Top of Atmosphere Radiance Product"
        # Setting the properties of Landsat image that will be needed to create descriptions
        col = stac.properties["landsat:collection_number"][1]
        lvl = stac.properties["landsat:correction"][1]
        sfx = "Top of Atmosphere Radiance" if lvl == "1" else "Surface Reflectance"
        # Getting dtype
        dtype = get_dtype(mtl, col)
        # Reading projection data for a panchromatic band
        if rsp_type in ["Landsat_olitirs_up_l1", "Landsat_etm_up_l1"]:
            pan_sr_shape = get_sr_shape(mtl, scene, pan=True)
            pan_transform = get_transform(pan_sr_shape, scene, bbox)
        else:
            pan_sr_shape = None
            pan_transform = None

        b1, b2, b3, b4, b5, b6, b61, b62, b7, b8, b9, b10, b11 = [None] * 13
        qa, qar, qaa, qat, qac, qaao = [None] * 6
        trad, urad, drad, atran, emis, emsd, cdist = [None] * 7

        # Getting bands if an MTL file is available
        if mtl is not None:
            gsd_ref = int(float(mtl.find(".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_REFLECTIVE").text))
            if "Landsat_olitirs" in rsp_type or "Landsat_etm" in rsp_type:
                gsd_pan = int(float(mtl.find(".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_PANCHROMATIC").text))
            else:
                gsd_pan = None
            if "Landsat_olitirs" in rsp_type or "Landsat_etm" in rsp_type or "Landsat_tm" in rsp_type:
                gsd_the = int(float(mtl.find(".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_THERMAL").text))
            else:
                gsd_the = None
            if "Landsat_olitirs" in rsp_type:
                b1 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_1").text
                b2 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_2").text
                b3 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_3").text
                b4 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_4").text
                b5 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_5").text
                b6 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_6").text
                b7 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_7").text
                b8 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_8").text if lvl == "1" else None
                b9 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_9").text if lvl == "1" else None
                b10 = (
                    mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_10").text
                    if lvl == "1"
                    else mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_ST_B10").text
                )
                b11 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_11").text if lvl == "1" else None
                qa = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L1_PIXEL").text
                qar = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION").text
                qaa = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L2_AEROSOL").text if lvl == "2" else None
                qat = (
                    mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L2_SURFACE_TEMPERATURE").text
                    if lvl == "2"
                    else None
                )
                trad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_THERMAL_RADIANCE").text if lvl == "2" else None
                drad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_DOWNWELL_RADIANCE").text if lvl == "2" else None
                urad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_UPWELL_RADIANCE").text if lvl == "2" else None
                atran = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_ATMOSPHERIC_TRANSMITTANCE").text if lvl == "2" else None
                emis = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_EMISSIVITY").text if lvl == "2" else None
                emsd = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_EMISSIVITY_STDEV").text if lvl == "2" else None
                cdist = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_CLOUD_DISTANCE").text if lvl == "2" else None
            elif "Landsat_etm" in rsp_type:
                b1 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_1").text
                b2 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_2").text
                b3 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_3").text
                b4 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_4").text
                b5 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_5").text
                b61 = (
                    mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_6_VCID_1").text
                    if lvl == "1"
                    else mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_ST_B6").text
                )
                b62 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_6_VCID_2").text if lvl == "1" else None
                b7 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_7").text
                b8 = mtl.get("PRODUCT_CONTENTS/FILE_NAME_BAND_8").text if lvl == "1" else None
                qa = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L1_PIXEL").text
                qar = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION").text
                qaao = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_ATMOSPHERIC_OPACITY").text if lvl == "2" else None
                qac = (
                    mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L2_SURFACE_REFLECTANCE_CLOUD").text
                    if lvl == "2"
                    else None
                )
                qat = (
                    mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L2_SURFACE_TEMPERATURE").text
                    if lvl == "2"
                    else None
                )
                trad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_THERMAL_RADIANCE").text if lvl == "2" else None
                drad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_DOWNWELL_RADIANCE").text if lvl == "2" else None
                urad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_UPWELL_RADIANCE").text if lvl == "2" else None
                atran = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_ATMOSPHERIC_TRANSMITTANCE").text if lvl == "2" else None
                emis = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_EMISSIVITY").text if lvl == "2" else None
                emsd = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_EMISSIVITY_STDEV").text if lvl == "2" else None
                cdist = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_CLOUD_DISTANCE").text if lvl == "2" else None
            elif "Landsat_tm" in rsp_type:
                b1 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_1").text
                b2 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_2").text
                b3 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_3").text
                b4 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_4").text
                b5 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_5").text
                b6 = (
                    mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_6").text
                    if lvl == "1"
                    else mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_ST_B6").text
                )
                b7 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_7").text
                qa = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L1_PIXEL").text
                qar = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION").text
                qaao = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_ATMOSPHERIC_OPACITY").text if lvl == "2" else None
                qac = (
                    mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L2_SURFACE_REFLECTANCE_CLOUD").text
                    if lvl == "2"
                    else None
                )
                qat = (
                    mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L2_SURFACE_TEMPERATURE").text
                    if lvl == "2"
                    else None
                )
                trad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_THERMAL_RADIANCE").text if lvl == "2" else None
                drad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_DOWNWELL_RADIANCE").text if lvl == "2" else None
                urad = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_UPWELL_RADIANCE").text if lvl == "2" else None
                atran = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_ATMOSPHERIC_TRANSMITTANCE").text if lvl == "2" else None
                emis = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_EMISSIVITY").text if lvl == "2" else None
                emsd = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_EMISSIVITY_STDEV").text if lvl == "2" else None
                cdist = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_CLOUD_DISTANCE").text if lvl == "2" else None
            elif "Landsat_mss" in rsp_type:
                if ("LM01" in name) or ("LM02" in name) or ("LM03" in name):
                    b1 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_4").text
                    b2 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_5").text
                    b3 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_6").text
                    b4 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_7").text
                else:
                    b1 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_1").text
                    b2 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_2").text
                    b3 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_3").text
                    b4 = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_BAND_4").text
                qa = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L1_PIXEL").text
                qar = mtl.find(".//PRODUCT_CONTENTS/FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION").text
        else:
            # Getting bands if an MTL file is not available
            gsd_ref = 60 if "Landsat_mss" in rsp_type else 30
            gsd_pan = 15 if "Landsat_olitirs" in rsp_type or "Landsat_etm" in rsp_type else None
            if "Landsat_olitirs" in rsp_type:
                gsd_the = 100
            elif "Landsat_etm" in rsp_type:
                gsd_the = 60
            elif "Landsat_tm" in rsp_type:
                gsd_the = 120
            else:
                gsd_the = None

        # Adding bands to dataset
        if "Landsat_olitirs" in rsp_type:
            if b1 is not None:
                add_asset(
                    item=stac,
                    name="coastal",
                    path=b1 if b1 else "B1.TIF",
                    common_name="coastal",
                    title="Coastal/Aerosol Band (B1)",
                    description="Collection " + col + " Level-" + lvl + " Coastal/Aerosol Band (B1) " + sfx,
                    bname="B1",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.443,
                    nodata=0,
                    dtype=dtype,
                )
            if b2 is not None:
                add_asset(
                    item=stac,
                    name="blue",
                    path=b2 if b2 else "B2.TIF",
                    common_name="blue",
                    title="Blue Band (B2)",
                    description="Collection " + col + " Level-" + lvl + " Blue Band (B2) " + sfx,
                    bname="B2",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.482,
                    nodata=0,
                    dtype=dtype,
                )
            if b3 is not None:
                add_asset(
                    item=stac,
                    name="green",
                    path=b3 if b3 else "B3.TIF",
                    common_name="green",
                    title="Green Band (B3)",
                    description="Collection " + col + " Level-" + lvl + " Green Band (B3) " + sfx,
                    bname="B3",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.561,
                    nodata=0,
                    dtype=dtype,
                )
            if b4 is not None:
                add_asset(
                    item=stac,
                    name="red",
                    path=b4 if b4 else "B4.TIF",
                    common_name="red",
                    title="Red Band (B4)",
                    description="Collection " + col + " Level-" + lvl + " Red Band (B4) " + sfx,
                    bname="B4",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.654,
                    nodata=0,
                    dtype=dtype,
                )
            if b5 is not None:
                add_asset(
                    item=stac,
                    name="nir08",
                    path=b5 if b5 else "B5.TIF",
                    common_name="nir08",
                    title="Near Infrared Band 0.8 (B5)",
                    description="Collection " + col + " Level-" + lvl + " Near Infrared Band 0.8 (B5) " + sfx,
                    bname="B5",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.865,
                    nodata=0,
                    dtype=dtype,
                )
            if b6 is not None:
                add_asset(
                    item=stac,
                    name="swir16",
                    path=b6 if b6 else "B6.TIF",
                    common_name="swir16",
                    title="Short-wave Infrared Band 1.6 (B6)",
                    description="Collection " + col + " Level-" + lvl + " Short-wave Infrared Band 1.6 (B6) " + sfx,
                    bname="B6",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=1.608,
                    nodata=0,
                    dtype=dtype,
                )
            if b7 is not None:
                add_asset(
                    item=stac,
                    name="swir22",
                    path=b7 if b7 else "B7.TIF",
                    common_name="swir22",
                    title="Short-wave Infrared Band 2.2 (B7)",
                    description="Collection " + col + " Level-" + lvl + " Short-wave Infrared Band 2.2 (B7) " + sfx,
                    bname="B7",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=2.2,
                    nodata=0,
                    dtype=dtype,
                )
            if lvl == "1":
                if b8 is not None:
                    add_asset(
                        item=stac,
                        name="pan",
                        path=b8 if b8 else "B8.TIF",
                        common_name="pan",
                        title="Panchromatic Band (B8)",
                        description="Collection " + col + " Level-" + lvl + " Panchromatic Band (B8) " + sfx,
                        bname="B8",
                        roles=["data"],
                        gsd=gsd_pan,
                        center_wavelength=0.589,
                        nodata=0,
                        dtype=dtype,
                        shape=pan_sr_shape,
                        transform=pan_transform,
                    )
                if b9 is not None:
                    add_asset(
                        item=stac,
                        name="cirrus",
                        path=b9 if b9 else "B9.TIF",
                        common_name="cirrus",
                        title="Cirrus Band (B9)",
                        description="Collection " + col + " Level-" + lvl + " Cirrus Band (B9) " + sfx,
                        bname="B9",
                        roles=["data"],
                        gsd=gsd_ref,
                        center_wavelength=1.373,
                        nodata=0,
                        dtype=dtype,
                    )
            if b10 is not None:
                add_asset(
                    item=stac,
                    name="lwir11",
                    path=b10 if b10 else "B10.TIF",
                    common_name="lwir11",
                    title="Thermal Infrared Band 10.9 (B10)",
                    description="Collection " + col + " Level-" + lvl + " Thermal Infrared Band 10.9 (B10) " + sfx,
                    bname="B10",
                    roles=["data", "temperature"],
                    gsd=gsd_the,
                    center_wavelength=10.888,
                    nodata=0,
                    dtype=dtype,
                )
            if lvl == "1" and b11 is not None:
                add_asset(
                    item=stac,
                    name="lwir12",
                    path=b11 if b11 else "B11.TIF",
                    common_name="lwir12",
                    title="Thermal Infrared Band 12.0 (B11)",
                    description="Collection " + col + " Level-" + lvl + " Thermal Infrared Band 12.0 (B11) " + sfx,
                    bname="B11",
                    roles=["data", "temperature"],
                    gsd=gsd_the,
                    center_wavelength=12.01,
                    nodata=0,
                    dtype=dtype,
                )
            if qa is not None:
                add_asset(
                    item=stac,
                    name="qa_pixel",
                    path=qa if qa else "QA_PIXEL.TIF",
                    common_name="qa_pixel",
                    title="Pixel Quality Assessment Band",
                    description="Collection " + col + " Level-" + lvl + " Pixel Quality Assessment Band " + sfx,
                    bname="QA",
                    roles=["cloud", "cloud-shadow", "snow-ice", "water-mask"],
                )
            if qar is not None:
                add_asset(
                    item=stac,
                    name="qa_radsat",
                    path=qar if qar else "QA_RADSAT.TIF",
                    common_name="qa_radsat",
                    title="Radiometric Saturation Quality Assessment Band",
                    description="Collection "
                    + col
                    + " Level-"
                    + lvl
                    + " Radiometric Saturation Quality Assessment Band "
                    + sfx,
                    bname="QA_RADSAT",
                    roles=["saturation"],
                )
            if lvl == "2" and qaa is not None:
                add_asset(
                    item=stac,
                    name="qa_aerosol",
                    path=qaa if qaa else "QA_AEROSOL.TIF",
                    common_name="qa_aerosol",
                    title="Aerosol Quality Analysis Band",
                    description="Collection " + col + " Level-" + lvl + " Aerosol Quality Analysis Band " + sfx,
                    bname="QA_AEROSOL",
                    roles=["metadata", "data-mask", "water-mask"],
                )
            if lvl == "2" and qat is not None:
                add_asset(
                    item=stac,
                    name="qa",
                    path=qat if qat else "ST_QA.TIF",
                    common_name="qa",
                    title="Surface Temperature Quality Assessment Band",
                    description="Collection "
                    + col
                    + " Level-"
                    + lvl
                    + " Surface Temperature Quality Assessment Band "
                    + sfx,
                    bname="ST_QA",
                    roles=["data"],
                )
            if lvl == "2" and trad is not None:
                add_asset(
                    item=stac,
                    name="TRAD",
                    path=trad if trad else "TRAD.TIF",
                    common_name="TRAD",
                    title="Thermal Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Thermal Radiance Band " + sfx,
                    bname="TRAD",
                    roles=["data"],
                )
            if lvl == "2" and drad is not None:
                add_asset(
                    item=stac,
                    name="DRAD",
                    path=drad if drad else "DRAD.TIF",
                    common_name="DRAD",
                    title="Downwelled Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Downwelled Radiance Band " + sfx,
                    bname="DRAD",
                    roles=["data"],
                )
            if lvl == "2" and urad is not None:
                add_asset(
                    item=stac,
                    name="URAD",
                    path=urad if urad else "URAD.TIF",
                    common_name="URAD",
                    title="Upwelled Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Upwelled Radiance Band " + sfx,
                    bname="URAD",
                    roles=["data"],
                )
            if lvl == "2" and atran is not None:
                add_asset(
                    item=stac,
                    name="ATRAN",
                    path=atran if atran else "ATRAN.TIF",
                    common_name="ATRAN",
                    title="Atmospheric Transmittance Band",
                    description="Collection " + col + " Level-" + lvl + " Atmospheric Transmittance Band " + sfx,
                    bname="ATRAN",
                    roles=["data"],
                )
            if lvl == "2" and emis is not None:
                add_asset(
                    item=stac,
                    name="EMIS",
                    path=qar if qar else "EMIS.TIF",
                    common_name="EMIS",
                    title="Emissivity Band",
                    description="Collection " + col + " Level-" + lvl + " Emissivity Band " + sfx,
                    bname="EMIS",
                    roles=["data"],
                )
            if lvl == "2" and emsd is not None:
                add_asset(
                    item=stac,
                    name="EMSD",
                    path=emsd if emsd else "EMSD.TIF",
                    common_name="EMSD",
                    title="Emissivity Standard Deviation Band",
                    description="Collection " + col + " Level-" + lvl + " Emissivity Standard Deviation Band " + sfx,
                    bname="EMSD",
                    roles=["data"],
                )
            if lvl == "2" and cdist is not None:
                add_asset(
                    item=stac,
                    name="CDIST",
                    path=cdist if cdist else "CDIST.TIF",
                    common_name="CDIST",
                    title="Cloud Distance Band",
                    description="Collection " + col + " Level-" + lvl + " Cloud Distance Band " + sfx,
                    bname="CDIST",
                    roles=["data"],
                )
        elif "Landsat_etm" in rsp_type:
            if b1 is not None:
                add_asset(
                    item=stac,
                    name="blue",
                    path=b1 if b1 else "B1.TIF",
                    common_name="blue",
                    title="Blue Band (B1)",
                    description="Collection " + col + " Level-" + lvl + " Blue Band (B1) " + sfx,
                    bname="B1",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.477,
                    nodata=0,
                    dtype=dtype,
                )
            if b2 is not None:
                add_asset(
                    item=stac,
                    name="green",
                    path=b2 if b2 else "B2.TIF",
                    common_name="green",
                    title="Green Band (B2)",
                    description="Collection " + col + " Level-" + lvl + " Green Band (B2) " + sfx,
                    bname="B2",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.56,
                    nodata=0,
                    dtype=dtype,
                )
            if b3 is not None:
                add_asset(
                    item=stac,
                    name="red",
                    path=b3 if b3 else "B3.TIF",
                    common_name="red",
                    title="Red Band (B3)",
                    description="Collection " + col + " Level-" + lvl + " Red Band (B3) " + sfx,
                    bname="B3",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.661,
                    nodata=0,
                    dtype=dtype,
                )
            if b4 is not None:
                add_asset(
                    item=stac,
                    name="nir08",
                    path=b4 if b4 else "B4.TIF",
                    common_name="nir08",
                    title="Near Infrared Band 0.8 (B4)",
                    description="Collection " + col + " Level-" + lvl + " Near Infrared Band 0.8 (B4) " + sfx,
                    bname="B4",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.835,
                    nodata=0,
                    dtype=dtype,
                )
            if b5 is not None:
                add_asset(
                    item=stac,
                    name="swir16",
                    path=b5 if b5 else "B5.TIF",
                    common_name="swir16",
                    title="Short-wave Infrared Band 1.6 (B5)",
                    description="Collection " + col + " Level-" + lvl + " Short-wave Infrared Band 1.6 (B5) " + sfx,
                    bname="B5",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=1.648,
                    nodata=0,
                    dtype=dtype,
                )
            if b61 is not None:
                add_asset(
                    item=stac,
                    name="lwir11",
                    path=b61 if b61 else "B6_VCID_1.TIF" if lvl == "1" else "B6.TIF",
                    common_name="lwir11",
                    title="Thermal Infrared Band 11.4 (B6)",
                    description="Collection " + col + " Level-" + lvl + " Thermal Infrared Band 11.4 (B6) " + sfx,
                    bname="B6_VCID_1" if lvl == "1" else "B6",
                    roles=["data", "temperature"],
                    gsd=gsd_the,
                    center_wavelength=11.45,
                    nodata=0,
                    dtype=dtype,
                )
            if lvl == "1" and b62 is not None:
                add_asset(
                    item=stac,
                    name="lwir12",
                    path=b62 if b62 else "B6_VCID_2.TIF",
                    common_name="lwir12",
                    title="Thermal Infrared Band (B6) high gain",
                    description="Collection " + col + " Level-" + lvl + " Thermal Infrared Band (B6) high gain " + sfx,
                    bname="B6_VCID_2",
                    roles=["data", "temperature"],
                    gsd=gsd_the,
                    center_wavelength=11.45,
                    nodata=0,
                    dtype=dtype,
                )
            if b7 is not None:
                add_asset(
                    item=stac,
                    name="swir22",
                    path=b7 if b7 else "B7.TIF",
                    common_name="swir22",
                    title="Short-wave Infrared Band 2.2 (B7)",
                    description="Collection " + col + " Level-" + lvl + " Short-wave Infrared Band 2.2 (B7) " + sfx,
                    bname="B7",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=2.204,
                    nodata=0,
                    dtype=dtype,
                )
            if lvl == "1" and b8 is not None:
                add_asset(
                    item=stac,
                    name="pan",
                    path=b8 if b8 else "B8.TIF",
                    common_name="pan",
                    title="Panchromatic Band (B8)",
                    description="Collection " + col + " Level-" + lvl + " Panchromatic Band (B8) " + sfx,
                    bname="B8",
                    roles=["data"],
                    gsd=gsd_pan,
                    center_wavelength=0.71,
                    nodata=0,
                    dtype=dtype,
                    shape=pan_sr_shape,
                    transform=pan_transform,
                )
            if qa is not None:
                add_asset(
                    item=stac,
                    name="qa_pixel",
                    path=qa if qa else "QA_PIXEL.TIF",
                    common_name="qa_pixel",
                    title="Pixel Quality Assessment Band",
                    description="Collection " + col + " Level-" + lvl + " Pixel Quality Assessment Band " + sfx,
                    bname="QA",
                    roles=["cloud", "cloud-shadow", "snow-ice", "water-mask"],
                )
            if qar is not None:
                add_asset(
                    item=stac,
                    name="qa_radsat",
                    path=qar if qar else "QA_RADSAT.TIF",
                    common_name="qa_radsat",
                    title="Radiometric Saturation Quality Assessment Band",
                    description="Collection "
                    + col
                    + " Level-"
                    + lvl
                    + " Radiometric Saturation Quality Assessment Band "
                    + sfx,
                    bname="QA_RADSAT",
                    roles=["saturation"],
                )
            if lvl == "2" and qaao is not None:
                add_asset(
                    item=stac,
                    name="atmos_opacity",
                    path=qaao if qaao else "ATMOS_OPACITY.TIF",
                    common_name="atmos_opacity",
                    title="Atmospheric Opacity Band",
                    description="Collection " + col + " Level-" + lvl + " Atmospheric Opacity Band " + sfx,
                    bname="ATMOS_OPACITY",
                    roles=["data"],
                )
            if lvl == "2" and qac is not None:
                add_asset(
                    item=stac,
                    name="cloud_qa",
                    path=qac if qac else "CLOUD_QA.TIF",
                    common_name="cloud_qa",
                    title="Cloud Quality Analysis Band",
                    description="Collection " + col + " Level-" + lvl + " Cloud Quality Analysis Band " + sfx,
                    bname="CLOUD_QA",
                    roles=["metadata", "cloud", "cloud-shadow", "snow-ice", "water-mask"],
                )
            if lvl == "2" and qat is not None:
                add_asset(
                    item=stac,
                    name="qa",
                    path=qat if qat else "ST_QA.TIF",
                    common_name="qa",
                    title="Surface Temperature Quality Assessment Band",
                    description="Collection "
                    + col
                    + " Level-"
                    + lvl
                    + " Surface Temperature Quality Assessment Band "
                    + sfx,
                    bname="ST_QA",
                    roles=["data"],
                )
            if lvl == "2" and trad is not None:
                add_asset(
                    item=stac,
                    name="TRAD",
                    path=qar if qar else "TRAD.TIF",
                    common_name="TRAD",
                    title="Thermal Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Thermal Radiance Band " + sfx,
                    bname="TRAD",
                    roles=["data"],
                )
            if lvl == "2" and drad is not None:
                add_asset(
                    item=stac,
                    name="DRAD",
                    path=qar if qar else "DRAD.TIF",
                    common_name="DRAD",
                    title="Downwelled Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Downwelled Radiance Band " + sfx,
                    bname="DRAD",
                    roles=["data"],
                )
            if lvl == "2" and urad is not None:
                add_asset(
                    item=stac,
                    name="URAD",
                    path=qar if qar else "URAD.TIF",
                    common_name="URAD",
                    title="Upwelled Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Upwelled Radiance Band " + sfx,
                    bname="URAD",
                    roles=["data"],
                )
            if lvl == "2" and atran is not None:
                add_asset(
                    item=stac,
                    name="ATRAN",
                    path=qar if qar else "ATRAN.TIF",
                    common_name="ATRAN",
                    title="Atmospheric Transmittance Band",
                    description="Collection " + col + " Level-" + lvl + " Atmospheric Transmittance Band " + sfx,
                    bname="ATRAN",
                    roles=["data"],
                )
            if lvl == "2" and emis is not None:
                add_asset(
                    item=stac,
                    name="EMIS",
                    path=qar if qar else "EMIS.TIF",
                    common_name="EMIS",
                    title="Emissivity Band",
                    description="Collection " + col + " Level-" + lvl + " Emissivity Band " + sfx,
                    bname="EMIS",
                    roles=["data"],
                )
            if lvl == "2" and emsd is not None:
                add_asset(
                    item=stac,
                    name="EMSD",
                    path=qar if qar else "EMSD.TIF",
                    common_name="EMSD",
                    title="Emissivity Standard Deviation Band",
                    description="Collection " + col + " Level-" + lvl + " Emissivity Standard Deviation Band " + sfx,
                    bname="EMSD",
                    roles=["data"],
                )
            if lvl == "2" and cdist is not None:
                add_asset(
                    item=stac,
                    name="CDIST",
                    path=qar if qar else "CDIST.TIF",
                    common_name="CDIST",
                    title="Cloud Distance Band",
                    description="Collection " + col + " Level-" + lvl + " Cloud Distance Band " + sfx,
                    bname="CDIST",
                    roles=["data"],
                )
        elif "Landsat_tm" in rsp_type:
            if b1 is not None:
                add_asset(
                    item=stac,
                    name="blue",
                    path=b1 if b1 else "B1.TIF",
                    common_name="blue",
                    title="Blue Band (B1)",
                    description="Collection " + col + " Level-" + lvl + " Blue Band (B1) " + sfx,
                    bname="B1",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.48,
                    nodata=0,
                    dtype=dtype,
                )
            if b2 is not None:
                add_asset(
                    item=stac,
                    name="green",
                    path=b2 if b2 else "B2.TIF",
                    common_name="green",
                    title="Green Band (B2)",
                    description="Collection " + col + " Level-" + lvl + " Green Band (B2) " + sfx,
                    bname="B2",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.56,
                    nodata=0,
                    dtype=dtype,
                )
            if b3 is not None:
                add_asset(
                    item=stac,
                    name="red",
                    path=b3 if b3 else "B3.TIF",
                    common_name="red",
                    title="Red Band (B3)",
                    description="Collection " + col + " Level-" + lvl + " Red Band (B3) " + sfx,
                    bname="B3",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.66,
                    nodata=0,
                    dtype=dtype,
                )
            if b4 is not None:
                add_asset(
                    item=stac,
                    name="nir08",
                    path=b4 if b4 else "B4.TIF",
                    common_name="nir08",
                    title="Near Infrared Band 0.8 (B4)",
                    description="Collection " + col + " Level-" + lvl + " Near Infrared Band 0.8 (B4) " + sfx,
                    bname="B4",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.83,
                    nodata=0,
                    dtype=dtype,
                )
            if b5 is not None:
                add_asset(
                    item=stac,
                    name="swir16",
                    path=b5 if b5 else "B5.TIF",
                    common_name="swir16",
                    title="Short-wave Infrared Band 1.6 (B5)",
                    description="Collection " + col + " Level-" + lvl + " Short-wave Infrared Band 1.6 (B5) " + sfx,
                    bname="B5",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=1.65,
                    nodata=0,
                    dtype=dtype,
                )
            if b6 is not None:
                add_asset(
                    item=stac,
                    name="lwir11",
                    path=b6 if b6 else "B6.TIF",
                    common_name="lwir11",
                    title="Thermal Infrared Band 11.4 (B6)",
                    description="Collection " + col + " Level-" + lvl + " Thermal Infrared Band 11.4 (B6) " + sfx,
                    bname="B6",
                    roles=["data", "temperature"],
                    gsd=gsd_the,
                    center_wavelength=11.45,
                    nodata=0,
                    dtype=dtype,
                )
            if b7 is not None:
                add_asset(
                    item=stac,
                    name="swir22",
                    path=b7 if b7 else "B7.TIF",
                    common_name="swir22",
                    title="Short-wave Infrared Band 2.2 (B7)",
                    description="Collection " + col + " Level-" + lvl + " Short-wave Infrared Band 2.2 (B7) " + sfx,
                    bname="B7",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=2.22,
                    nodata=0,
                    dtype=dtype,
                )
            if qa is not None:
                add_asset(
                    item=stac,
                    name="qa_pixel",
                    path=qa if qa else "QA_PIXEL.TIF",
                    common_name="qa_pixel",
                    title="Pixel Quality Assessment Band",
                    description="Collection " + col + " Level-" + lvl + " Pixel Quality Assessment Band " + sfx,
                    bname="QA",
                    roles=["cloud", "cloud-shadow", "snow-ice", "water-mask"],
                )
            if qar is not None:
                add_asset(
                    item=stac,
                    name="qa_radsat",
                    path=qar if qar else "QA_RADSAT.TIF",
                    common_name="qa_radsat",
                    title="Radiometric Saturation Quality Assessment Band",
                    description="Collection "
                    + col
                    + " Level-"
                    + lvl
                    + " Radiometric Saturation Quality Assessment Band "
                    + sfx,
                    bname="QA_RADSAT",
                    roles=["saturation"],
                )
            if lvl == "2" and qaao is not None:
                add_asset(
                    item=stac,
                    name="atmos_opacity",
                    path=qaao if qaao else "ATMOS_OPACITY.TIF",
                    common_name="atmos_opacity",
                    title="Atmospheric Opacity Band",
                    description="Collection " + col + " Level-" + lvl + " Atmospheric Opacity Band " + sfx,
                    bname="ATMOS_OPACITY",
                    roles=["data"],
                )
            if lvl == "2" and qac is not None:
                add_asset(
                    item=stac,
                    name="cloud_qa",
                    path=qac if qac else "CLOUD_QA.TIF",
                    common_name="cloud_qa",
                    title="Cloud Quality Analysis Band",
                    description="Collection " + col + " Level-" + lvl + " Cloud Quality Analysis Band " + sfx,
                    bname="CLOUD_QA",
                    roles=["metadata", "cloud", "cloud-shadow", "snow-ice", "water-mask"],
                )
            if lvl == "2" and qat is not None:
                add_asset(
                    item=stac,
                    name="qa",
                    path=qat if qat else "ST_QA.TIF",
                    common_name="qa",
                    title="Surface Temperature Quality Assessment Band",
                    description="Collection "
                    + col
                    + " Level-"
                    + lvl
                    + " Surface Temperature Quality Assessment Band "
                    + sfx,
                    bname="ST_QA",
                    roles=["data"],
                )
            if lvl == "2" and trad is not None:
                add_asset(
                    item=stac,
                    name="TRAD",
                    path=qar if qar else "TRAD.TIF",
                    common_name="TRAD",
                    title="Thermal Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Thermal Radiance Band " + sfx,
                    bname="TRAD",
                    roles=["data"],
                )
            if lvl == "2" and drad is not None:
                add_asset(
                    item=stac,
                    name="DRAD",
                    path=qar if qar else "DRAD.TIF",
                    common_name="DRAD",
                    title="Downwelled Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Downwelled Radiance Band " + sfx,
                    bname="DRAD",
                    roles=["data"],
                )
            if lvl == "2" and urad is not None:
                add_asset(
                    item=stac,
                    name="URAD",
                    path=qar if qar else "URAD.TIF",
                    common_name="URAD",
                    title="Upwelled Radiance Band",
                    description="Collection " + col + " Level-" + lvl + " Upwelled Radiance Band " + sfx,
                    bname="URAD",
                    roles=["data"],
                )
            if lvl == "2" and atran is not None:
                add_asset(
                    item=stac,
                    name="ATRAN",
                    path=qar if qar else "ATRAN.TIF",
                    common_name="ATRAN",
                    title="Atmospheric Transmittance Band",
                    description="Collection " + col + " Level-" + lvl + " Atmospheric Transmittance Band " + sfx,
                    bname="ATRAN",
                    roles=["data"],
                )
            if lvl == "2" and emis is not None:
                add_asset(
                    item=stac,
                    name="EMIS",
                    path=qar if qar else "EMIS.TIF",
                    common_name="EMIS",
                    title="Emissivity Band",
                    description="Collection " + col + " Level-" + lvl + " Emissivity Band " + sfx,
                    bname="EMIS",
                    roles=["data"],
                )
            if lvl == "2" and emsd is not None:
                add_asset(
                    item=stac,
                    name="EMSD",
                    path=qar if qar else "EMSD.TIF",
                    common_name="EMSD",
                    title="Emissivity Standard Deviation Band",
                    description="Collection " + col + " Level-" + lvl + " Emissivity Standard Deviation Band " + sfx,
                    bname="EMSD",
                    roles=["data"],
                )
            if lvl == "2" and cdist is not None:
                add_asset(
                    item=stac,
                    name="CDIST",
                    path=qar if qar else "CDIST.TIF",
                    common_name="CDIST",
                    title="Cloud Distance Band",
                    description="Collection " + col + " Level-" + lvl + " Cloud Distance Band " + sfx,
                    bname="CDIST",
                    roles=["data"],
                )
        elif "Landsat_mss" in rsp_type:
            if b1 is not None:
                title = "Green Band (B1)" if ("LM04" in name) or ("LM05" in name) else "Green Band (B4)"
                add_asset(
                    item=stac,
                    name="green",
                    path=b1 if b1 else "B1.TIF" if ("LM04" in name) or ("LM05" in name) else "B4.TIF",
                    common_name="green",
                    title=title,
                    description="Collection " + col + " Level-" + lvl + " " + title + " " + sfx,
                    bname="B1" if ("LM04" in name) or ("LM05" in name) else "B4",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.55,
                    nodata=0,
                    dtype=dtype,
                )
            if b2 is not None:
                title = "Red Band (B2)" if ("LM04" in name) or ("LM05" in name) else "Red Band (B5)"
                add_asset(
                    item=stac,
                    name="red",
                    path=b2 if b2 else "B2.TIF" if ("LM04" in name) or ("LM05" in name) else "B5.TIF",
                    common_name="red",
                    title=title,
                    description="Collection " + col + " Level-" + lvl + " " + title + " " + sfx,
                    bname="B2" if ("LM04" in name) or ("LM05" in name) else "B5",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.65,
                    nodata=0,
                    dtype=dtype,
                )
            if b3 is not None:
                title = (
                    "Near Infrared Band 0.7 (B3)"
                    if ("LM04" in name) or ("LM05" in name)
                    else "Near Infrared Band 0.7 (B6)"
                )
                add_asset(
                    item=stac,
                    name="nir08",
                    path=b3 if b3 else "B3.TIF" if ("LM04" in name) or ("LM05" in name) else "B6.TIF",
                    common_name="nir08",
                    title=title,
                    description="Collection " + col + " Level-" + lvl + " " + title + " " + sfx,
                    bname="B3" if ("LM04" in name) or ("LM05" in name) else "B6",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.75,
                    nodata=0,
                    dtype=dtype,
                )
            if b4 is not None:
                title = (
                    "Near Infrared Band 0.9 (B4)"
                    if ("LM04" in name) or ("LM05" in name)
                    else "Near Infrared Band 0.9 (B7)"
                )
                add_asset(
                    item=stac,
                    name="nir09",
                    path=b4 if b4 else "B4.TIF" if ("LM04" in name) or ("LM05" in name) else "B7.TIF",
                    common_name="nir09",
                    title=title,
                    description="Collection " + col + " Level-" + lvl + " " + title + " " + sfx,
                    bname="B4" if ("LM04" in name) or ("LM05" in name) else "B7",
                    roles=["data"],
                    gsd=gsd_ref,
                    center_wavelength=0.95,
                    nodata=0,
                    dtype=dtype,
                )
            if qa is not None:
                add_asset(
                    item=stac,
                    name="qa_pixel",
                    path=qa if qa else "QA_PIXEL.TIF",
                    common_name="qa_pixel",
                    title="Pixel Quality Assessment Band",
                    description="Collection " + col + " Level-" + lvl + " Pixel Quality Assessment Band " + sfx,
                    bname="QA",
                    roles=["cloud", "cloud-shadow", "snow-ice", "water-mask"],
                )
            if qar is not None:
                add_asset(
                    item=stac,
                    name="qa_radsat",
                    path=qar if qar else "QA_RADSAT.TIF",
                    common_name="qa_radsat",
                    title="Radiometric Saturation Quality Assessment Band",
                    description="Collection "
                    + col
                    + " Level-"
                    + lvl
                    + " Radiometric Saturation Quality Assessment Band "
                    + sfx,
                    bname="QA_RADSAT",
                    roles=["saturation"],
                )

    # Fix geometries if needed
    stac = antimeridian.fix_item(stac, antimeridian.Strategy.SPLIT)
    filter_bands(stac, LANDSAT_BANDS + PAN_BAND)

    # Validation
    validate(stac)
    if rsp_type not in rsp_landsat_types:
        raise ValueError("Failed to read Landsat product.")
    return stac


def postprocess_landsat_dataset(
    dataset: pystac.Item,
    img: Dataset,
    json_path: Path,
    pansharpen: bool,
    keep_pan_band: bool,
    pan: DataArray,
) -> None:
    """Postprocess Landsat STAC."""
    # Rewriting datatype
    if dataset.common_metadata.description in ["Landsat_mss_up_l1", "Landsat_mss_up_l2"]:
        add_rsp_metadata(dataset, rsp_type="Landsat_mss_p")
    elif dataset.common_metadata.description in ["Landsat_tm_up_l1", "Landsat_tm_up_l2"]:
        add_rsp_metadata(dataset, rsp_type="Landsat_tm_p")
    elif dataset.common_metadata.description in ["Landsat_etm_up_l1", "Landsat_etm_up_l2"]:
        add_rsp_metadata(dataset, rsp_type="Landsat_etm_p")
    elif dataset.common_metadata.description in ["Landsat_olitirs_up_l1", "Landsat_olitirs_up_l2"]:
        add_rsp_metadata(dataset, rsp_type="Landsat_olitirs_p")

    # Deleting unneeded assets
    filter_bands(dataset, (LANDSAT_BANDS + PAN_BAND) if keep_pan_band else LANDSAT_BANDS)

    # Adding hrefs
    for asset in dataset.assets:
        dataset.assets[asset].href = dataset.assets[asset].ext.eo.bands[0].name + ".tif"

    # Changing datatypes
    for asset in dataset.assets:
        if dataset.assets[asset].ext.has("raster") and dataset.assets[asset].ext.raster.bands is not None:
            if pansharpen is False and keep_pan_band is True and asset == "pan":
                dataset.assets[asset].ext.raster.bands[0].data_type = pan.dtype.name
            else:
                dataset.assets[asset].ext.raster.bands[0].data_type = img[asset].dtype.name

    # Updating projection info
    for asset in dataset.assets:
        if pansharpen is False and keep_pan_band is True and asset == "pan":
            dataset.assets[asset].ext.proj.shape = list(pan.shape[-2:])
            dataset.assets[asset].ext.proj.transform = list(pan.rio.transform())[:6]
    dataset.ext.proj.shape = list(img.red.shape[-2:])
    dataset.ext.proj.transform = list(img.red.rio.transform())[:6]
    dataset.ext.proj.epsg = img.red.rio.crs.to_epsg()
    # noinspection PyTypeChecker
    dataset.bbox = list(transform_bounds(img.red.rio.crs, "EPSG:4326", *img.red.rio.bounds()))
    dataset.geometry["type"] = "Polygon"
    dataset.geometry["coordinates"] = [
        [
            [dataset.bbox[0], dataset.bbox[1]],  # LL
            [dataset.bbox[2], dataset.bbox[1]],  # UL
            [dataset.bbox[2], dataset.bbox[3]],  # UR
            [dataset.bbox[0], dataset.bbox[3]],  # LR
            [dataset.bbox[0], dataset.bbox[1]],  # LL
        ],
    ]

    # Adding self link
    dataset.set_self_href(json_path.as_posix())

    # Fix geometries if needed
    dataset = antimeridian.fix_item(dataset, antimeridian.Strategy.SPLIT)
    # Validation
    validate(dataset)
