from pydantic import AnyUrl, TypeAdapter
from typing import Union

import re
import warnings
import xml.etree.ElementTree as ET  # noqa: N817
import zipfile

import dateutil
import requests

import rasterio as rio
from rasterio.warp import transform_bounds

import pystac
from antimeridian import FixWindingWarning
from stactools.core.utils import antimeridian

from remote_sensing_processor.common.common_functions import ping
from remote_sensing_processor.common.dataset import (
    add_asset,
    add_rsp_metadata,
    etree_to_dict,
    filter_bands,
    validate,
)
from remote_sensing_processor.common.types import FilePath
from remote_sensing_processor.imagery.types import get_type, rsp_sentinel2_types


SENTINEL2_BANDS = [
    "coastal",
    "blue",
    "green",
    "red",
    "rededge071",
    "rededge075",
    "rededge078",
    "nir",
    "nir08",
    "nir09",
    "cirrus",
    "swir16",
    "swir22",
]
QA_BANDS = ["scl", "cldmsk", "cldprb"]

SENTINEL2_EXTENSION_SCHEMA = "https://stac-extensions.github.io/sentinel-2/v1.0.0/schema.json"

class MTD:
    def __init__(self, bands, tl=False):
        self.mtd = None
        adapter = TypeAdapter(Union[AnyUrl, FilePath])
        for band in bands:
            if tl:
                b = re.search("MTD_TL.xml", str(band))
            else:
                b = re.search(r"MTD_MSIL\d\w.xml", str(band))
            if b is not None:
                if isinstance(band, str):
                    band = adapter.validate_python(band)
                    if isinstance(band, AnyUrl):
                        if ping(band):
                            r = requests.get(band)
                            xml = ET.fromstring(r.content)
                            self.mtd = etree_to_dict(xml)
                            self.root = None
                    else:
                        xml = ET.parse(band).getroot()
                        self.mtl = etree_to_dict(xml)
                        self.root = band.parent
                else:
                    if ".zip" in str(band):
                        for parent in band.parents:
                            if ".zip" in parent.suffixes:
                                with zipfile.ZipFile(parent) as zip:
                                    file = zip.open(band.relative_to(parent).as_posix())
                                    xml = ET.parse(file).getroot()
                    else:
                        xml = ET.parse(band).getroot()
                    self.mtd = etree_to_dict(xml)
                    self.root = band.parent
        # Additional data processing
        if self.mtd is not None:
            # Getting rid of the root
            self.mtd = self.mtd[list(self.mtd.keys())[0]]

    def get(self, values: list):
        if self.mtd is not None:
            el = self.mtd
            for key in values:
                el = el[key]
            return el
        else:
            raise ValueError("MTD metadata file not found")


def get_bands(path):
    """Read Sentinel-2 bands from directory or archive."""
    if isinstance(path, pystac.Item):
        bands = [path.assets[i].href for i in path.assets]
        archive = False
    else:
        if path.is_file() and (".zip" in path.suffixes):
            archive = True
            with zipfile.ZipFile(path) as file:
                bands = file.namelist()
                for i in range(len(bands)):
                    bands[i] = path / bands[i]
        elif path.is_dir():
            archive = False
            bands = list(path.glob("**/*"))
        else:
            raise ValueError(str(path) + " is not a directory or zip archive")
    return bands, archive


def prepare_band(band, archive):
    """Converts pathlib.Path into a convenient string."""
    if archive:
        return "zip://" + band.as_posix()
    else:
        return band.as_posix()


def add_raster_info(dataset, mtd):
    """Add raster band to existing STAC."""
    for asset in dataset.assets:
        nodata = 0
        if mtd.mtd is not None:
            if asset == "scl":
                lst = mtd.get([
                    "General_Info",
                    "Product_Image_Characteristics",
                    "Scene_Classification_List",
                    "Scene_Classification_ID"]
                )
                for i in lst:
                    if i["SCENE_CLASSIFICATION_TEXT"] == "SC_NODATA":
                        nodata = int(i["SCENE_CLASSIFICATION_INDEX"])
                        break
            else:
                lst = mtd.get(["General_Info", "Product_Image_Characteristics", "Special_Values"])
                for i in lst:
                    if i["SPECIAL_VALUE_TEXT"] == "NODATA":
                        nodata = int(i["SPECIAL_VALUE_INDEX"])
                        break
        dtype = "uint16"
        raster = pystac.extensions.raster.RasterExtension.ext(dataset.assets[asset], add_if_missing=True)
        raster.bands = [
            pystac.extensions.raster.RasterBand.create(
                nodata=nodata,
                data_type=dtype,
            )
        ]


def get_name(path, bands, mtd):
    """Get Sentinel 2 name to be used as ID (e.g. S2A_MSIL1C_20240624T074621_N0510_R135_T38TMK_20240624T083616.SAFE)."""
    # Trying to read from STAC
    if isinstance(path, pystac.Item) and "s2:product_uri" in path.properties:
        return path.properties["s2:product_uri"]
    # Trying to read from mtd
    if mtd is not None and mtd.mtd is not None:
        return mtd.get(["General_Info", "Product_Info", "PRODUCT_URI"])
    # Else trying to read from path
    sentinel2_re = r"S\d\w_MSIL\d\w_\d\d\d\d\d\d\d\d\w\d\d\d\d\d\d_\w\d\d\d\d_\w\d\d\d_\w\d\d\w\w\w_\d\d\d\d\d\d\d\d\w\d\d\d\d\d\d.SAFE"
    p = re.search(sentinel2_re, path.stem)
    if p:
        return p.group(0)
    # Else trying to find in band names
    for band in bands:
        p = re.search(sentinel2_re, band.stem)
        if p:
            return p.group(0)
    # Else just return path name
    return path.stem


def get_datetime(name, mtd):
    """
    Generate datetime from MTD (more precise) or from name (less precise).

    20240624T074621
    012345678901234
    2024-06-24T07:46:21.024Z
    """
    if mtd.mtd is not None:
        time = mtd.get(["General_Info", "Product_Info", "PRODUCT_START_TIME"])
    else:
        if re.search(r"_\d\d\d\d\d\d\d\dT\d\d\d\d\d\d_", name):
            date = re.search(r"_\d\d\d\d\d\d\d\dT\d\d\d\d\d\d_", name).group(0).strip("_")
        else:
            # No date in the name
            date = "20000101T000000"
        time = date[0:4] + "-" + date[4:6] + "-" + date[6:11] + ":" + date[11:13] + ":" + date[13:15] + ".000Z"
    return dateutil.parser.parse(time)


def get_bbox(mtd, bands, archive):
    """Get bounding box from MTD or rasterio bbox."""
    if mtd.mtd is not None:
        lats = []
        lons = []
        for i, c in enumerate(
            mtd.get(
                [
                    "Geometric_Info",
                    "Product_Footprint",
                    "Product_Footprint",
                    "Global_Footprint",
                    "EXT_POS_LIST",
                ]
            ).split(" ")
        ):
            if i % 2 == 0:
                lats.append(float(c))
            else:
                lons.append(float(c))
        return [min(lons), min(lats), max(lons), max(lats)]
    else:
        band = [band for band in bands if ".jp2" in band.suffixes or ".tif" in band.suffixes][0]
        if archive:
            band = "zip://" + band.as_posix()
        with rio.open(band) as bnd:
            return list(transform_bounds(bnd.crs, "EPSG:4326", *bnd.bounds))


def get_coords(bbox, mtd):
    """Get coordinates from ANG file or bbox."""
    if mtd.mtd is not None:
        lats = []
        lons = []
        for i, c in enumerate(
            mtd.get(
                [
                    "Geometric_Info",
                    "Product_Footprint",
                    "Product_Footprint",
                    "Global_Footprint",
                    "EXT_POS_LIST",
                ]
            ).split(" ")
        ):
            if i % 2 == 0:
                lats.append(float(c))
            else:
                lons.append(float(c))
        coordinates = [
            [
                [lons[4], lats[4]],
                [lons[3], lats[3]],
                [lons[2], lats[2]],
                [lons[1], lats[1]],
                [lons[0], lats[0]],
            ]
        ]
    else:
        # Just construct it from bbox
        coordinates = [
            [
                [bbox[0], bbox[1]],  # LL
                [bbox[0], bbox[3]],  # LR
                [bbox[2], bbox[3]],  # UR
                [bbox[2], bbox[1]],  # UL
                [bbox[0], bbox[1]],  # LL
            ]
        ]
    return {"type": "Polygon", "coordinates": coordinates}


def get_cloud_cover(mtd):
    if mtd.mtd is not None:
        return float(mtd.get(["Quality_Indicators_Info", "Cloud_Coverage_Assessment"]))
    else:
        return 0.0


def get_epsg(mtd_tl, bands, archive):
    """Get CRS from MTL or one of the bands."""
    if mtd_tl.mtd is not None:
        return int(mtd_tl.get(["Geometric_Info", "Tile_Geocoding", "HORIZONTAL_CS_CODE"]).split(":")[1])
    else:
        # Read from one of the bands
        band = [band for band in bands if ".jp2" in band.suffixes or ".tif" in band.suffixes][0]
        if archive:
            band = "zip://" + band.as_posix()
        with rio.open(band) as bnd:
            return bnd.crs.to_epsg()


def get_sr_shape(mtd_tl, bands, archive, gsd):
    """Getting image shape from MTL or one of the bands."""
    assert gsd in [10, 20, 60]
    if mtd_tl.mtd is not None:
        for s in mtd_tl.get(["Geometric_Info", "Tile_Geocoding", "Size"]):
            if int(s["@resolution"]) == gsd:
                return [int(s["NROWS"]), int(s["NCOLS"])]
    else:
        if gsd == 10:
            band = [
                band
                for band in bands
                if (".jp2" in band.suffixes or ".tif" in band.suffixes) and ("B08" in band.name or "B8" in band.name)
            ][0]
        elif gsd == 20:
            band = [
                band
                for band in bands
                if (".jp2" in band.suffixes or ".tif" in band.suffixes) and ("B07" in band.name or "B7" in band.name)
            ][0]
        elif gsd == 60:
            band = [
                band
                for band in bands
                if (".jp2" in band.suffixes or ".tif" in band.suffixes) and ("B09" in band.name or "B9" in band.name)
            ][0]
        if archive:
            band = "zip://" + band.as_posix()
        with rio.open(band) as bnd:
            return [bnd.height, bnd.width]


def get_transform(bands, archive, gsd):
    assert gsd in [10, 20, 60]
    if gsd == 10:
        band = [
            band
            for band in bands
            if (".jp2" in band.suffixes or ".tif" in band.suffixes) and ("B08" in band.name or "B8" in band.name)
        ][0]
    elif gsd == 20:
        band = [
            band
            for band in bands
            if (".jp2" in band.suffixes or ".tif" in band.suffixes) and ("B07" in band.name or "B7" in band.name)
        ][0]
    elif gsd == 60:
        band = [
            band
            for band in bands
            if (".jp2" in band.suffixes or ".tif" in band.suffixes)
            and ("B09" in band.name or "B9" in band.name)
        ][0]
    if archive:
        band = "zip://" + band.as_posix()
    with rio.open(band) as bnd:
        return list(bnd.transform)[:6]


def read_sentinel2_dataset(path):
    warnings.filterwarnings("ignore", category=FixWindingWarning)

    # Reading bands
    bands, archive = get_bands(path)

    # Reading MTD
    mtd = MTD(bands)
    # Reading MTD_TL
    mtd_tl = MTD(bands, tl=True)

    # Reading sentinel2 name
    name = get_name(path, bands, mtd)

    # Getting RSP type
    type = get_type(path)
    if "Undefined" in type:
        type = get_type(name)

    if isinstance(path, pystac.Item):
        stac = path
        rename = {
            "coastal": ["B01", "B1", "B01.tif", "B1.tif", "B01.jp2", "B1.jp2"],
            "blue": ["B02", "B2", "B02.tif", "B2.tif", "B02.jp2", "B2.jp2"],
            "green": ["B03", "B3", "B03.tif", "B3.tif", "B03.jp2", "B3.jp2"],
            "red": ["B04", "B4", "B04.tif", "B4.tif", "B04.jp2", "B4.jp2"],
            "rededge071": ["B05", "B5", "B05.tif", "B5.tif", "B05.jp2", "B5.jp2"],
            "rededge075": ["B06", "B6", "B06.tif", "B6.tif", "B06.jp2", "B6.jp2"],
            "rededge078": ["B07", "B7", "B07.tif", "B7.tif", "B07.jp2", "B7.jp2"],
            "nir": ["B08", "B8", "B08.tif", "B8.tif", "B08.jp2", "B8.jp2"],
            "nir08": ["B8A", "B8A.tif", "B8A.jp2"],
            "nir09": ["B09", "B9", "B09.tif", "B9.tif", "B09.jp2", "B9.jp2"],
            "swir16": ["B11", "B11.tif", "B11.jp2"],
            "swir22": ["B12", "B12.tif", "B12.jp2"],
            "scl": ["SCL", "SCL.tif", "scl.tif", "SCL.jp2", "scl.jp2"],
            "cldmsk": ["CLDMSK", "CLD_MSK", "MSK_CLOUDS_B00.gml", "MSK_CLASSI_B00.jp2"],
            "cldprb": ["CLDPRB", "CLD_PRB", "MSK_CLDPRB_20m.jp2"],
        }
        for aname in list(stac.assets.keys()):
            for destname, names in rename.items():
                if aname in names and destname not in stac.assets:
                    stac.assets[destname] = stac.assets.pop(aname)
                    break

        # Setting up type
        if type == "Undefined":
            type = get_type(stac.id)
        add_rsp_metadata(stac, rsp_type=type)

        # Some of the metadata is not relevant and will break the stac
        stac.clear_links()
        stac.collection_id = None

        # Removing unneeded assets and add dtype and nodata info
        filter_bands(stac, SENTINEL2_BANDS + QA_BANDS)
        add_raster_info(stac, mtd)

    else:
        # Getting datetime
        datetime = get_datetime(name, mtd)

        # Getting bounding box
        bbox = get_bbox(mtd, bands, archive)
        # Getting coords
        coords = get_coords(bbox, mtd)

        # Creating a STAC item
        stac = pystac.Item(
            id=name,
            datetime=datetime,
            geometry=coords,
            bbox=bbox,
            properties={},
        )
        add_rsp_metadata(stac, rsp_type=type)

        # Adding EO extension
        eo_ext = pystac.extensions.eo.EOExtension.ext(stac, add_if_missing=True)
        cloud_cover = get_cloud_cover(mtd)
        eo_ext.cloud_cover = cloud_cover

        # Adding projection info
        proj_ext = pystac.extensions.projection.ProjectionExtension.ext(stac, add_if_missing=True)
        proj_ext.epsg = get_epsg(mtd_tl, bands, archive)
        proj_ext.shape = get_sr_shape(mtd_tl, bands, archive, 10)
        shape_20 = get_sr_shape(mtd_tl, bands, archive, 20)
        shape_60 = get_sr_shape(mtd_tl, bands, archive, 60)
        proj_ext.transform = get_transform(bands, archive, 10)
        transform_20 = get_transform(bands, archive, 20)
        transform_60 = get_transform(bands, archive, 60)

        # Adding visual info
        if mtd_tl.mtd is not None:
            view_ext = pystac.extensions.view.ViewExtension.ext(stac, add_if_missing=True)
            view_ext.sun_azimuth = float(
                mtd_tl.get(
                    [
                        "Geometric_Info",
                        "Tile_Angles",
                        "Mean_Sun_Angle",
                        "AZIMUTH_ANGLE",
                        "#text",
                    ]
                )
            )
            view_ext.sun_elevation = 90 - float(
                mtd_tl.get(
                    [
                        "Geometric_Info",
                        "Tile_Angles",
                        "Mean_Sun_Angle",
                        "ZENITH_ANGLE",
                        "#text",
                    ]
                )
            )

        # Writing common metadata
        if mtd.mtd is not None:
            stac.common_metadata.platform = mtd.get(["General_Info", "Product_Info", "Datatake", "SPACECRAFT_NAME"])
        elif name.startswith("S2A"):
            stac.common_metadata.platform = "Sentinel-2A"
        elif name.startswith("S2B"):
            stac.common_metadata.platform = "Sentinel-2B"
        else:
            stac.common_metadata.platform = "Sentinel-2C"
        stac.common_metadata.gsd = 10
        stac.common_metadata.instruments = ["msi"]

        # Adding Sentinel-2 extension
        stac.stac_extensions.append(SENTINEL2_EXTENSION_SCHEMA)
        d = {}
        d["s2:product_uri"] = name
        if mtd.mtd is not None:
            d["s2:tile_id"] = mtd.get(
                [
                    "General_Info",
                    "Product_Info",
                    "Product_Organisation",
                    "Granule_List",
                    "Granule",
                    "@granuleIdentifier",
                ]
            )
            d["s2:datatake_id"] = mtd.get(["General_Info", "Product_Info", "Datatake", "@datatakeIdentifier"])
            d["s2:datastrip_id"] = mtd.get(
                [
                    "General_Info",
                    "Product_Info",
                    "Product_Organisation",
                    "Granule_List",
                    "Granule",
                    "@datastripIdentifier",
                ]
            )
            d["s2:datatake_type"] = mtd.get(["General_Info", "Product_Info", "Datatake", "DATATAKE_TYPE"])
            d["s2:reflectance_conversion_factor"] = float(
                mtd.get(
                    [
                        "General_Info",
                        "Product_Image_Characteristics",
                        "Reflectance_Conversion",
                        "U",
                    ]
                )
            )
            d["s2:degraded_msi_data_percentage"] = float(
                mtd.get(
                    [
                        "Quality_Indicators_Info",
                        "Technical_Quality_Assessment",
                        "DEGRADED_MSI_DATA_PERCENTAGE",
                    ]
                )
            )
            if type == "Sentinel2_up_l2":
                d["s2:water_percentage"] = float(
                    mtd.get(["Quality_Indicators_Info", "Image_Content_QI", "WATER_PERCENTAGE"])
                )
                d["s2:vegetation_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "VEGETATION_PERCENTAGE",
                        ]
                    )
                )
                d["s2:thin_cirrus_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "THIN_CIRRUS_PERCENTAGE",
                        ]
                    )
                )
                d["s2:cloud_shadow_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "CLOUD_SHADOW_PERCENTAGE",
                        ]
                    )
                )
                d["s2:nodata_pixel_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "NODATA_PIXEL_PERCENTAGE",
                        ]
                    )
                )
                d["s2:unclassified_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "UNCLASSIFIED_PERCENTAGE",
                        ]
                    )
                )
                try:
                    d["s2:dark_features_percentage"] = float(
                        mtd.get(
                            [
                                "Quality_Indicators_Info",
                                "Image_Content_QI",
                                "CAST_SHADOW_PERCENTAGE",
                            ]
                        )
                    )
                except:
                    d["s2:dark_features_percentage"] = float(
                        mtd.get(
                            [
                                "Quality_Indicators_Info",
                                "Image_Content_QI",
                                "DARK_FEATURES_PERCENTAGE",
                            ]
                        )
                    )
                d["s2:not_vegetated_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "NOT_VEGETATED_PERCENTAGE",
                        ]
                    )
                )
                d["s2:high_proba_clouds_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "HIGH_PROBA_CLOUDS_PERCENTAGE",
                        ]
                    )
                )
                d["s2:medium_proba_clouds_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "MEDIUM_PROBA_CLOUDS_PERCENTAGE",
                        ]
                    )
                )
                d["s2:saturated_defective_pixel_percentage"] = float(
                    mtd.get(
                        [
                            "Quality_Indicators_Info",
                            "Image_Content_QI",
                            "SATURATED_DEFECTIVE_PIXEL_PERCENTAGE",
                        ]
                    )
                )
        stac.properties.update(**d)

        b1, b2, b3, b4, b5, b6, b7, b8, b8a, b9, b10, b11, b12, scl, cldmsk, cldprb = [None] * 16

        # Getting band pathes
        if mtd.mtd is not None:
            band_pathes = mtd.get(
                [
                    "General_Info",
                    "Product_Info",
                    "Product_Organisation",
                    "Granule_List",
                    "Granule",
                    "IMAGE_FILE",
                ]
            )
            for band in band_pathes:
                if ("B01" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("60m" in band))):
                    b1 = mtd.root / (band + ".jp2")
                elif ("B02" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("10m" in band))):
                    b2 = mtd.root / (band + ".jp2")
                elif ("B03" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("10m" in band))):
                    b3 = mtd.root / (band + ".jp2")
                elif ("B04" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("10m" in band))):
                    b4 = mtd.root / (band + ".jp2")
                elif ("B05" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band))):
                    b5 = mtd.root / (band + ".jp2")
                elif ("B06" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band))):
                    b6 = mtd.root / (band + ".jp2")
                elif ("B07" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band))):
                    b7 = mtd.root / (band + ".jp2")
                elif ("B08" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("10m" in band))):
                    b8 = mtd.root / (band + ".jp2")
                elif ("B8A" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band))):
                    b8a = mtd.root / (band + ".jp2")
                elif ("B09" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("60m" in band))):
                    b9 = mtd.root / (band + ".jp2")
                elif ("B10" in band) and (type == "Sentinel2_up_l1"):
                    b10 = mtd.root / (band + ".jp2")
                elif ("B11" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band))):
                    b11 = mtd.root / (band + ".jp2")
                elif ("B12" in band) and ((type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band))):
                    b12 = mtd.root / (band + ".jp2")
                elif ("SCL_20m" in band) and (type == "Sentinel2_up_l2"):
                    scl = mtd.root / (band + ".jp2")
        else:
            for band in bands:
                if ("B01" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("60m" in band.name))
                ):
                    b1 = band
                elif ("B02" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("10m" in band.name))
                ):
                    b2 = band
                elif ("B03" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("10m" in band.name))
                ):
                    b3 = band
                elif ("B04" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("10m" in band.name))
                ):
                    b4 = band
                elif ("B05" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band.name))
                ):
                    b5 = band
                elif ("B06" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band.name))
                ):
                    b6 = band
                elif ("B07" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band.name))
                ):
                    b7 = band
                elif ("B08" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("10m" in band.name))
                ):
                    b8 = band
                elif ("B8A" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band.name))
                ):
                    b8a = band
                elif ("B09" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("60m" in band.name))
                ):
                    b9 = band
                elif ("B10" in band.name) and (type == "Sentinel2_up_l1"):
                    b10 = band
                elif ("B11" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band.name))
                ):
                    b11 = band
                elif ("B12" in band.name) and (
                    (type == "Sentinel2_up_l1") or ((type == "Sentinel2_up_l2") and ("20m" in band.name))
                ):
                    b12 = band
                elif ("SCL_20m" in band.name) and (type == "Sentinel2_up_l2"):
                    scl = band
        for band in bands:
            if "MSK_CLOUDS_B00.gml" in band.name:
                cldmsk = band
            elif "MSK_CLASSI_B00.jp2" in band.name:
                cldmsk = band
            elif ("MSK_CLDPRB_20m.jp2" in band.name) and (type == "Sentinel2_up_l2"):
                cldprb = band

        # Adding bands to dataset
        if b1 is not None:
            add_asset(
                item=stac,
                name="coastal",
                path=prepare_band(b1, archive),
                common_name="coastal",
                title="Band 1 - Coastal aerosol",
                description="Band 1 - Coastal aerosol",
                bname="B01",
                roles=["data"],
                gsd=60,
                center_wavelength=0.443,
                nodata=0,
                dtype="uint16",
                shape=shape_60,
                transform=transform_60,
            )
        if b2 is not None:
            add_asset(
                item=stac,
                name="blue",
                path=prepare_band(b2, archive),
                common_name="blue",
                title="Band 2 - Blue",
                description="Band 2 - Blue",
                bname="B02",
                roles=["data"],
                gsd=10,
                center_wavelength=0.49,
                nodata=0,
                dtype="uint16",
            )
        if b3 is not None:
            add_asset(
                item=stac,
                name="green",
                path=prepare_band(b3, archive),
                common_name="green",
                title="Band 3 - Green",
                description="Band 3 - Green",
                bname="B03",
                roles=["data"],
                gsd=10,
                center_wavelength=0.56,
                nodata=0,
                dtype="uint16",
            )
        if b4 is not None:
            add_asset(
                item=stac,
                name="red",
                path=prepare_band(b4, archive),
                common_name="red",
                title="Band 4 - Red",
                description="Band 4 - Red",
                bname="B04",
                roles=["data"],
                gsd=10,
                center_wavelength=0.665,
                nodata=0,
                dtype="uint16",
            )
        if b5 is not None:
            add_asset(
                item=stac,
                name="rededge071",
                path=prepare_band(b5, archive),
                common_name="rededge",
                title="Band 5 - Vegetation red edge 1",
                description="Band 5 - Vegetation red edge 1",
                bname="B05",
                roles=["data"],
                gsd=20,
                center_wavelength=0.704,
                nodata=0,
                dtype="uint16",
                shape=shape_20,
                transform=transform_20,
            )
        if b6 is not None:
            add_asset(
                item=stac,
                name="rededge075",
                path=prepare_band(b6, archive),
                common_name="rededge",
                title="Band 6 - Vegetation red edge 2",
                description="Band 6 - Vegetation red edge 2",
                bname="B06",
                roles=["data"],
                gsd=20,
                center_wavelength=0.74,
                nodata=0,
                dtype="uint16",
                shape=shape_20,
                transform=transform_20,
            )
        if b7 is not None:
            add_asset(
                item=stac,
                name="rededge078",
                path=prepare_band(b7, archive),
                common_name="rededge",
                title="Band 7 - Vegetation red edge 3",
                description="Band 7 - Vegetation red edge 3",
                bname="B07",
                roles=["data"],
                gsd=20,
                center_wavelength=0.783,
                nodata=0,
                dtype="uint16",
                shape=shape_20,
                transform=transform_20,
            )
        if b8 is not None:
            add_asset(
                item=stac,
                name="nir",
                path=prepare_band(b8, archive),
                common_name="nir",
                title="Band 8 - NIR",
                description="Band 8 - NIR",
                bname="B08",
                roles=["data"],
                gsd=10,
                center_wavelength=0.842,
                nodata=0,
                dtype="uint16",
            )
        if b8a is not None:
            add_asset(
                item=stac,
                name="nir08",
                path=prepare_band(b8a, archive),
                common_name="nir08",
                title="Band 8A - Vegetation red edge 4",
                description="Band 8A - Vegetation red edge 4",
                bname="B8A",
                roles=["data"],
                gsd=20,
                center_wavelength=0.865,
                nodata=0,
                dtype="uint16",
                shape=shape_20,
                transform=transform_20,
            )
        if b9 is not None:
            add_asset(
                item=stac,
                name="nir09",
                path=prepare_band(b9, archive),
                common_name="nir09",
                title="Band 9 - Water vapor",
                description="Band 9 - Water vapor",
                bname="B09",
                roles=["data"],
                gsd=60,
                center_wavelength=0.945,
                nodata=0,
                dtype="uint16",
                shape=shape_60,
                transform=transform_60,
            )
        if (type == "Sentinel2_up_l1" or type == "Sentinel2_p") and b10 is not None:
            add_asset(
                item=stac,
                name="cirrus",
                path=prepare_band(b10, archive),
                common_name="cirrus",
                title="Band 10 - Cirrus",
                description="Band 10 - Cirrus",
                bname="B10",
                roles=["data"],
                gsd=60,
                center_wavelength=1.374,
                nodata=0,
                dtype="uint16",
                shape=shape_60,
                transform=transform_60,
            )
        if b11 is not None:
            add_asset(
                item=stac,
                name="swir16",
                path=prepare_band(b11, archive),
                common_name="swir16",
                title="Band 11 - SWIR (1.6)",
                description="Band 11 - SWIR (1.6)",
                bname="B11",
                roles=["data"],
                gsd=20,
                center_wavelength=1.61,
                nodata=0,
                dtype="uint16",
                shape=shape_20,
                transform=transform_20,
            )
        if b12 is not None:
            add_asset(
                item=stac,
                name="swir22",
                path=prepare_band(b12, archive),
                common_name="swir22",
                title="Band 12 - SWIR (2.2)",
                description="Band 12 - SWIR (2.2)",
                bname="B12",
                roles=["data"],
                gsd=20,
                center_wavelength=2.19,
                nodata=0,
                dtype="uint16",
                shape=shape_20,
                transform=transform_20,
            )
        if type == "Sentinel2_up_l2" and scl is not None:
            add_asset(
                item=stac,
                name="scl",
                path=prepare_band(scl, archive),
                title="Scene classfication map (SCL)",
                description="Scene classfication map (SCL)",
                bname="SCL",
                roles=["cloud"],
                gsd=20,
                center_wavelength=0,
                nodata=0,
                dtype="uint8",
                shape=shape_20,
                transform=transform_20,
            )
        if cldmsk is not None:
            add_asset(
                item=stac,
                name="cldmsk",
                path=prepare_band(cldmsk, archive),
                title="Cloud Mask",
                description="Cloud Mask",
                bname="MSK_CLOUDS",
                roles=["cloud"],
                gsd=20,
                center_wavelength=0,
                nodata=0,
                dtype="uint8",
                shape=shape_60,
                transform=transform_60,
            )
        if type == "Sentinel2_up_l2" and cldprb is not None:
            add_asset(
                item=stac,
                name="cldprb",
                path=prepare_band(cldprb, archive),
                title="Cloud Probability Mask",
                description="Cloud Probability Mask",
                bname="CLDPRB",
                roles=["cloud"],
                gsd=20,
                center_wavelength=0,
                nodata=0,
                dtype="uint8",
                shape=shape_20,
                transform=transform_20,
            )

    # Fix geometries if needed
    stac = antimeridian.fix_item(stac, antimeridian.Strategy.SPLIT)
    filter_bands(stac, SENTINEL2_BANDS + QA_BANDS)

    # Validation
    validate(stac)
    assert type in rsp_sentinel2_types
    return stac, mtd


def postprocess_sentinel2_dataset(dataset, imgs, json_path, upscale):
    # Rewriting datatype
    add_rsp_metadata(dataset, rsp_type="Sentinel2_p")

    # Deleting unneeded assets
    filter_bands(dataset, SENTINEL2_BANDS)

    # Adding hrefs
    for asset in dataset.assets:
        dataset.assets[asset].href = dataset.assets[asset].ext.eo.bands[0].name + ".tif"

    # Changing gsd
    if upscale is not None:
        for asset in dataset.assets:
            dataset.assets[asset].common_metadata.gsd = 10

    # Changing datatypes
    for i in imgs:
        for band in i:
            if (
                dataset.assets[band].ext.has("raster")
                and dataset.assets[band].ext.raster.bands is not None
            ):
                dataset.assets[band].ext.raster.bands[0].data_type = i[band].dtype.name

    # Updating projection info
    dataset.ext.proj.shape = list(imgs[0].red.shape[-2:])
    dataset.ext.proj.transform = list(imgs[0].red.rio.transform())[:6]
    dataset.ext.proj.epsg = imgs[0].red.rio.crs.to_epsg()
    dataset.bbox = list(transform_bounds(imgs[0].red.rio.crs, "EPSG:4326", *imgs[0].red.rio.bounds()))
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

    # Updating projection info for each asset
    if len(imgs) == 1:
        # Removing projection info if resample is not None
        for asset in dataset.assets:
            if dataset.assets[asset].ext.proj.shape is not None:
                dataset.assets[asset].ext.proj.shape = None
            if dataset.assets[asset].ext.proj.transform is not None:
                dataset.assets[asset].ext.proj.transform = None
    else:
        # Changing projection info if resample is not None
        for i in imgs:
            for band in i:
                if dataset.assets[band].ext.proj.shape is not None:
                    dataset.assets[band].ext.proj.shape = list(i[band].shape[-2:])
                if dataset.assets[band].ext.proj.transform is not None:
                    dataset.assets[band].ext.proj.transform = list(i[band].rio.transform())[:6]

    # Adding self link
    dataset.set_self_href(json_path)

    # Fix geometries if needed
    dataset = antimeridian.fix_item(dataset, antimeridian.Strategy.SPLIT)
    # Validation
    validate(dataset)
