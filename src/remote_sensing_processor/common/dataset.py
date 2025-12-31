"""Commonly used STAC dataset functions."""

from typing import Any, Literal, Optional, Union

import datetime
import json
import tarfile
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

from xarray import Dataset

import rasterio as rio
from rasterio.warp import transform_bounds

import pystac
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterBand, RasterExtension
from stactools.core.utils import antimeridian

from remote_sensing_processor.common.common_functions import ping


def read_dataset(path: Union[pystac.Item, Path]) -> pystac.Item:
    """Read any dataset, from STAC file, multiband dataset from a directory or single-band one file dataset."""
    from remote_sensing_processor.imagery.types import get_type, rsp_all_types

    if isinstance(path, pystac.Item) or ".json" in path.suffixes:
        if isinstance(path, pystac.Item):
            stac = path.clone()
        else:
            stac = read_json(path)
            # If hrefs are relative, make them absolute
            stac.make_asset_hrefs_absolute()
            if stac is None:
                raise ValueError("Unable to read STAC dataset")

        # Getting RSP type
        if stac.common_metadata.description not in rsp_all_types or "Undefined" in stac.common_metadata.description:
            rsp_type = get_type(stac.id)
            add_rsp_metadata(stac, rsp_type=rsp_type)

        # Some metadata is not relevant and will break the stac
        stac.clear_links()
        stac.collection_id = None
    else:
        if path.is_dir():
            bands = list(path.glob("**/*.*[!(zip|tar|tar.gz|aux.xml)*]"))
        elif path.is_file():
            bands = [path]
        else:
            raise ValueError("Cannot read " + str(path))

        # Getting name
        name = path.stem

        # Getting RSP type
        rsp_type = get_type(path)
        if rsp_type == "Undefined":
            rsp_type = get_type(name)

        # Getting datetime (last modified time)
        time = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc)

        # Validating bands and getting coord params
        bands, bbox, coords, crs, shape, transform = get_coord_params(bands)

        # Creating STAC item
        stac = pystac.Item(
            id=name,
            datetime=time,
            geometry=coords,
            bbox=bbox,
            properties={},
        )
        add_rsp_metadata(stac, rsp_type=rsp_type)

        # Adding projection info
        proj_ext = ProjectionExtension.ext(stac, add_if_missing=True)
        proj_ext.epsg = crs
        proj_ext.shape = shape
        proj_ext.transform = transform

        # Adding self href
        if path.is_dir():
            stac.set_self_href((path / (stac.id + ".json")).as_posix())
        elif path.is_file():
            stac.set_self_href((path.parent / (stac.id + ".json")).as_posix())

        # Adding bands
        for band in bands:
            add_asset(
                item=stac,
                name=band.stem,
                path=band.as_posix(),
            )

    # Fix geometries if needed
    stac = antimeridian.fix_item(stac, antimeridian.Strategy.SPLIT)

    # Validation
    validate(stac)
    return stac


def validate(stac: pystac.Item) -> pystac.Item:
    """Validate STAC."""
    try:
        if ping("https://stac-extensions.github.io/"):
            stac.validate()
        else:
            warnings.warn("Cannot retrieve STAC schemas. Validation aborted.", stacklevel=2)
    except Exception:
        warnings.warn("Validation failed. Is internet connection available?", stacklevel=2)
    return stac


def is_multiband(inp: pystac.Item) -> bool:
    """Check if STAC dataset contains multiple bands."""
    return len(inp.assets) > 1


def check_output(
    input_path: Union[pystac.Item, Path],
    output_path: Path,
    parent: Union[bool, Literal["auto"]] = False,
) -> Path:
    """Check if an output path is valid or set or fix it."""
    if output_path is None:
        if isinstance(input_path, pystac.Item):
            if output_path is None:
                raise ValueError("output_path must be set if input is a STAC Item")
        else:
            if parent is True:
                output_path = input_path.parent
            if parent is False:
                output_path = input_path
            if parent == "auto":
                output_path = input_path if input_path.is_dir() else input_path.parent
    return output_path


def get_coord_params(
    bands: list[Path],
) -> tuple[list[Path], list[float], dict, rio.crs.CRS, list[int], list[Union[int, float]]]:
    """Filter bands and get bounding box, coords, crs, shape and transform for a dataset."""
    validated_bands = []
    bboxes = []
    crses = []
    shapes = []
    transforms = []
    # Read data from each band
    for band in bands:
        try:
            with rio.open(band) as bnd:
                validated_bands.append(band)
                # noinspection PyTypeChecker
                bboxes.append(list(transform_bounds(bnd.crs, "EPSG:4326", *bnd.bounds)))
                crses.append(bnd.crs.to_epsg())
                shapes.append([bnd.height, bnd.width])
                transforms.append(list(bnd.transform)[:6])
        except Exception:  # noqa: S110
            pass
    # Assert the bands have the same metadata
    if bboxes.count(bboxes[0]) != len(bboxes):
        raise ValueError("Bboxes are not valid")
    bbox = bboxes[0]
    if crses.count(crses[0]) != len(crses):
        raise ValueError("CRSes are not valid")
    crs = crses[0]
    if shapes.count(shapes[0]) != len(shapes):
        raise ValueError("Shapes are not valid")
    shape = shapes[0]
    if transforms.count(transforms[0]) != len(transforms):
        raise ValueError("Transforms are not valid")
    transform = transforms[0]
    coordinates = [
        [
            [bbox[0], bbox[1]],  # LL
            [bbox[2], bbox[1]],  # UL
            [bbox[2], bbox[3]],  # UR
            [bbox[0], bbox[3]],  # LR
            [bbox[0], bbox[1]],  # LL
        ],
    ]
    coords = {"type": "Polygon", "coordinates": coordinates}
    return validated_bands, bbox, coords, crs, shape, transform


def add_rsp_metadata(item: pystac.Item, rsp_type: Optional[str] = None) -> None:
    """Adds RSP metadata to STAC item."""
    # TODO: create an RSP STAC extension
    # Temporarily writing RSP type into description
    if rsp_type is not None:
        item.common_metadata.description = rsp_type


def add_asset(
    item: pystac.Item,
    name: str,
    path: Union[Path, str],
    common_name: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    bname: Optional[str] = None,
    roles: Optional[list[str]] = None,
    gsd: Optional[Union[int, float]] = None,
    center_wavelength: Optional[Union[int, float]] = None,
    nodata: Optional[Union[int, float]] = None,
    dtype: Optional[str] = None,
    shape: Optional[list[int]] = None,
    transform: Optional[list[Union[int, float]]] = None,
) -> None:
    """Adds a band to STAC item.

    :param item: item
    :param name: blue
    :param path: path
    :param common_name: blue
    :param title: Blue Band (B1)
    :param description: Collection 2 Level-1 Blue Band (B1) Top of Atmosphere Radiance
    :param bname: B1
    :param roles: ['data']
    :param gsd: 30
    :param center_wavelength: 0.49
    :param nodata 0
    :param dtype uint16
    :param shape [13981 16141]
    :param transform [5.0, 0.0, 240892.5, 0.0, -15.0, -3091492.5]
    :return: None
    """
    if isinstance(path, Path):
        path = path.as_posix()
    asset_fields: dict[str, Any] = {
        "href": path,
        "media_type": str(pystac.media_type.MediaType.COG),
    }
    if title is not None:
        asset_fields["title"] = title
    if description is not None:
        asset_fields["description"] = description
    if roles is not None:
        asset_fields["roles"] = roles
    asset_dict = {
        "name": name,
        "asset_fields": asset_fields,
    }
    asset = pystac.Asset(**asset_dict["asset_fields"])
    asset.set_owner(item)
    eo = EOExtension.ext(asset, add_if_missing=True)
    eo.bands = [Band.create(name=name)]
    if nodata is not None or dtype is not None:
        raster = RasterExtension.ext(asset, add_if_missing=True)
        raster.bands = [RasterBand.create()]
    if shape is not None or transform is not None:
        proj = ProjectionExtension.ext(asset, add_if_missing=True)
        if shape is not None:
            proj.shape = shape
        if transform is not None:
            proj.transform = transform
    item.add_asset(name, asset)
    if common_name is not None:
        item.assets[name].ext.eo.bands[0].common_name = common_name
    if bname is not None:
        item.assets[name].ext.eo.bands[0].name = bname
    if gsd is not None:
        item.assets[name].common_metadata.gsd = gsd
    if center_wavelength is not None:
        item.assets[name].ext.eo.bands[0].center_wavelength = center_wavelength
    if nodata is not None:
        item.assets[name].ext.raster.bands[0].nodata = nodata
    if dtype is not None:
        item.assets[name].ext.raster.bands[0].data_type = dtype


def filter_bands(dataset: pystac.Item, bands: list[str]) -> None:
    """Deleting unneeded assets."""
    unneeded = []
    for asset in dataset.assets:
        if asset not in bands:
            unneeded.append(asset)
    for asset in unneeded:
        dataset.assets.pop(asset)


def read_json(json_path: Any) -> Optional[pystac.Item]:
    """Reads pystac JSON."""
    if hasattr(json_path, "close"):
        d = json.load(json_path)
        try:
            return pystac.Item.from_dict(d)
        except Exception:
            return None
    elif ".tar" in str(json_path):
        for parent in json_path.parents:
            if ".tar" in parent.suffixes:
                with tarfile.open(parent) as tar:
                    file = tar.extractfile(json_path.relative_to(parent).as_posix())
                    d = json.load(file)
                    try:
                        return pystac.Item.from_dict(d)
                    except Exception:
                        return None
    elif ".zip" in str(json_path):
        for parent in json_path.parents:
            if ".zip" in parent.suffixes:
                with zipfile.ZipFile(parent) as zf:
                    file = zf.open(json_path.relative_to(parent).as_posix())
                    d = json.load(file)
                    try:
                        return pystac.Item.from_dict(d)
                    except Exception:
                        return None
    elif ".json" in json_path.suffixes:
        try:
            return pystac.Item.from_file(json_path)
        except Exception:
            return None
    return None


def postprocess_dataset(
    dataset: pystac.Item,
    raster: Dataset,
    output_path: Path,
    bands: Optional[list[str]] = None,
) -> tuple[pystac.Item, Path]:
    """Prepare dataset to writing to file."""
    # Deleting unneeded assets
    if bands is not None:
        filter_bands(dataset, bands)

    mb = is_multiband(dataset)
    if mb:
        if ".json" in output_path.suffixes:
            dataset.id = output_path.stem
            output_path = output_path.parent
        else:
            if output_path.is_file() or len(output_path.suffixes) != 0:
                raise ValueError("Cannot write multiband dataset to a file. Specify a directory instead.")
    else:
        if len(dataset.assets) > 1:
            raise ValueError("Something went wrong in a multiband logic.")
        dataset.id = output_path.stem

    dataset.datetime = datetime.datetime.now()

    # Adding hrefs
    for asset in dataset.assets:
        if not mb and (output_path.is_file() or len(output_path.suffixes) != 0) and ".json" not in output_path.suffixes:
            dataset.assets[asset].href = output_path.name
        else:
            dataset.assets[asset].href = dataset.assets[asset].ext.eo.bands[0].name + ".tif"

    # Changing datatypes and nodata
    for asset in dataset.assets:
        if dataset.assets[asset].ext.has("raster") and dataset.assets[asset].ext.raster.bands is not None:
            if dataset.assets[asset].ext.raster.bands[0].data_type is not None:
                dataset.assets[asset].ext.raster.bands[0].data_type = raster[asset].dtype.name
            if dataset.assets[asset].ext.raster.bands[0].nodata is not None and raster[asset].rio.nodata is not None:
                dataset.assets[asset].ext.raster.bands[0].nodata = float(raster[asset].rio.nodata)

    # Updating projection info
    dataset.ext.proj.shape = list(raster[next(iter(raster.data_vars))].shape[-2:])
    dataset.ext.proj.transform = list(raster[next(iter(raster.data_vars))].rio.transform())[:6]
    dataset.ext.proj.epsg = raster[next(iter(raster.data_vars))].rio.crs.to_epsg()
    # noinspection PyTypeChecker
    dataset.bbox = list(
        transform_bounds(
            raster[next(iter(raster.data_vars))].rio.crs,
            "EPSG:4326",
            *raster[next(iter(raster.data_vars))].rio.bounds(),
        ),
    )
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
    a = None
    for asset in dataset.assets:
        # Getting asset if raster is multiband
        if asset in raster:
            a = asset
        else:
            for a in raster:
                # Explicitly convert to string
                a = str(a)

                if asset == a.split("/")[0]:
                    break

        if a is None:
            raise ValueError("Band " + str(asset) + " not found in raster.")

        # Changing projection info if resample is not None
        if dataset.assets[asset].ext.proj.shape is not None:
            dataset.assets[asset].ext.proj.shape = list(raster[a].shape[-2:])
        if dataset.assets[asset].ext.proj.transform is not None:
            dataset.assets[asset].ext.proj.transform = list(raster[a].rio.transform())[:6]

    # Adding self link
    if output_path is not None:
        if ".json" in output_path.suffixes:
            json_path = output_path
        elif output_path.is_dir() or len(output_path.suffixes) == 0:  # directory
            json_path = output_path / (dataset.id + ".json")
        else:
            json_path = output_path.parent / (dataset.id + ".json")
        dataset.clear_links()
        dataset.collection_id = None
        dataset.set_self_href(json_path.as_posix())
    else:
        json_path = Path(dataset.get_links("self")[0].href)

    # Fix geometries if needed
    dataset = antimeridian.fix_item(dataset, antimeridian.Strategy.SPLIT)
    # Validation
    validate(dataset)
    return dataset, json_path


def etree_to_dict(t: Any) -> Any:
    """Converts etree to dict."""
    # Removing xmlns links from tag
    _, _, t.tag = t.tag.rpartition("}")
    # Converting
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]["#text"] = text
        else:
            d[t.tag] = text
    return d


def lines_to_dict(text: Any) -> Any:
    """Converts lines from Landsat MTL txt to dict."""
    d = {}
    group_level_1 = None
    group_level_2 = None
    for line in list(filter(None, text)):
        if line == "END":
            break
        name, val = [s.strip(" \"'\n") for s in line.split("=")]
        if name == "GROUP":
            if group_level_1 is None:
                group_level_1 = val
                d[group_level_1] = {}
            elif group_level_1 is not None and group_level_2 is None:
                group_level_2 = val
                d[group_level_1][group_level_2] = {}
        elif name == "END_GROUP":
            if group_level_2 is not None:
                group_level_2 = None
            elif group_level_2 is None and group_level_1 is not None:
                group_level_1 = None
        else:
            d[group_level_1][group_level_2][name] = val
    return d
