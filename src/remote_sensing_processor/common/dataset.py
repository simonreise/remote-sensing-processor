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
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds

import pystac
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterBand, RasterExtension
from stactools.core.utils import antimeridian

from remote_sensing_processor.common.common_functions import ping


def _bbox_to_polygon(bbox: list[float]) -> dict:
    """Create a GeoJSON Polygon from a bbox [min_lon, min_lat, max_lon, max_lat].

    Coordinates follow RFC 7946 counter-clockwise winding order for exterior rings.
    """
    coordinates = [
        [
            [bbox[0], bbox[1]],  # Lower-left (SW)
            [bbox[2], bbox[1]],  # Lower-right (SE)
            [bbox[2], bbox[3]],  # Upper-right (NE)
            [bbox[0], bbox[3]],  # Upper-left (NW)
            [bbox[0], bbox[1]],  # Lower-left (SW) — closing
        ],
    ]
    return {"type": "Polygon", "coordinates": coordinates}


def _get_time(path: Path) -> datetime.datetime:
    """Get a meaningful datetime from the output path (last modified time), with UTC fallback."""
    try:
        if path.exists():
            return datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc)
    except Exception:  # noqa: S110
        pass
    return datetime.datetime.now(datetime.timezone.utc)


def read_dataset(path: Union[pystac.Item, Path]) -> pystac.Item:
    """Read any dataset, from STAC file, multiband dataset from a directory or single-band one file dataset."""
    from remote_sensing_processor.imagery.types import get_type, rsp_all_types

    if isinstance(path, pystac.Item) or ".json" in path.suffixes:
        if isinstance(path, pystac.Item):
            stac = path.clone()
        else:
            stac = read_json(path)
            if stac is None:
                raise ValueError("Unable to read STAC dataset")
            # If hrefs are relative, make them absolute
            stac.make_asset_hrefs_absolute()

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
        time = _get_time(path)

        # Validating bands and getting coord params
        (
            bands,
            band_names_map,
            band_nodata_map,
            band_dtypes_map,
            bbox,
            coords,
            crs,
            shape,
            transform,
        ) = get_coord_params(bands)

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
            band_names = band_names_map.get(band, [band.stem])
            band_nodata = band_nodata_map.get(band, None)
            band_dtypes = band_dtypes_map.get(band, None)
            add_asset(
                item=stac,
                name=band.stem,
                path=band.as_posix(),
                band_names=band_names,
                band_nodata=band_nodata,
                band_dtypes=band_dtypes,
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


def is_multi_asset(inp: pystac.Item) -> bool:
    """Check if STAC dataset contains multiple assets (i.e. multiple separate files)."""
    return len(inp.assets) > 1


def has_multiband_assets(inp: pystac.Item) -> bool:
    """Check if any asset in the STAC item contains multiple bands (multiband raster)."""
    for asset_key in inp.assets:
        asset = inp.assets[asset_key]
        try:
            if asset.ext.has("eo") and asset.ext.eo.bands is not None and len(asset.ext.eo.bands) > 1:
                return True
        except Exception:  # noqa: S110
            pass
    return False


def get_band_count(inp: pystac.Item) -> int:
    """Get total number of bands across all assets."""
    count = 0
    for asset_key in inp.assets:
        asset = inp.assets[asset_key]
        try:
            if asset.ext.has("eo") and asset.ext.eo.bands is not None:
                count += len(asset.ext.eo.bands)
            else:
                count += 1
        except Exception:
            count += 1
    return count


def check_output(
    input_path: Union[pystac.Item, Path],
    output_path: Path,
    parent: Union[bool, Literal["auto"]] = False,
) -> Path:
    """Check if an output path is valid or set or fix it."""
    if output_path is None:
        if isinstance(input_path, pystac.Item):
            raise ValueError("output_path must be set if input is a STAC Item")
        if parent is True:
            output_path = input_path.parent
        elif parent == "auto":
            output_path = input_path if input_path.is_dir() else input_path.parent
        else:
            output_path = input_path
    return output_path


def get_coord_params(
    bands: list[Path],
) -> tuple[
    list[Path],
    dict[Path, list[str]],
    dict[Path, list[Union[int, float, None]]],
    dict[Path, list[str]],
    list[float],
    dict,
    rio.crs.CRS,
    list[int],
    list[Union[int, float]],
]:
    """Filter bands and get bounding box, coords, crs, shape and transform for a dataset."""
    validated_bands = []
    band_names_map: dict[Path, list[str]] = {}
    band_nodata_map: dict[Path, list[Union[int, float, None]]] = {}
    band_dtypes_map: dict[Path, list[str]] = {}
    bboxes = []
    crses = []
    shapes = []
    transforms = []
    # Read data from each band
    for band in bands:
        try:
            with rio.open(band) as bnd:
                validated_bands.append(band)
                names = list(bnd.descriptions)
                for i in range(len(names)):
                    if names[i] is None:
                        if len(names) > 1:
                            names[i] = f"{band.stem}_{i}"
                        else:
                            names[i] = band.stem
                band_names_map[band] = names
                band_nodata_map[band] = (
                    list(bnd.nodatavals)
                    if getattr(bnd, "nodatavals", None) is not None
                    else [getattr(bnd, "nodata", None)] * bnd.count
                )
                band_dtypes_map[band] = (
                    [dt.name if hasattr(dt, "name") else str(dt) for dt in bnd.dtypes]
                    if getattr(bnd, "dtypes", None) is not None
                    else [None] * bnd.count
                )
                # noinspection PyTypeChecker
                bboxes.append(list(transform_bounds(bnd.crs, "EPSG:4326", *bnd.bounds)))
                crses.append(bnd.crs.to_epsg())
                shapes.append([bnd.height, bnd.width])
                transforms.append(list(bnd.transform)[:6])
        except RasterioIOError:
            pass

    if not validated_bands:
        raise ValueError("No valid raster bands found")

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
    coords = _bbox_to_polygon(bbox)
    return (
        validated_bands,
        band_names_map,
        band_nodata_map,
        band_dtypes_map,
        bbox,
        coords,
        crs,
        shape,
        transform,
    )


def add_rsp_metadata(item: pystac.Item, rsp_type: Optional[str] = None) -> None:
    """Adds RSP metadata to STAC item."""
    # TODO: create an RSP STAC extension
    # Temporarily writing RSP type into description
    if rsp_type is not None:
        item.common_metadata.description = rsp_type


def _to_list(val: Any, length: int) -> list[Any]:
    """Helper to convert a scalar, sequence, or None into a list of a given length."""
    if val is None:
        return [None] * length
    if isinstance(val, (list, tuple)):
        return list(val) + [None] * (length - len(val))
    return [val] * length


def add_asset(
    item: pystac.Item,
    name: str,
    path: Union[Path, str],
    title: Optional[str] = None,
    description: Optional[str] = None,
    roles: Optional[list[str]] = None,
    gsd: Optional[Union[int, float]] = None,
    shape: Optional[list[int]] = None,
    transform: Optional[list[Union[int, float]]] = None,
    band_names: Optional[Union[str, list[str]]] = None,
    band_common_names: Optional[Union[str, list[Optional[str]]]] = None,
    band_nodata: Optional[Union[int, float, list[Union[int, float]]]] = None,
    band_dtypes: Optional[Union[str, list[str]]] = None,
    band_center_wavelengths: Optional[Union[int, float, list[Union[int, float]]]] = None,
) -> None:
    """Adds a band asset to a STAC item.

    Supports both single-band and multi-band raster assets using unified parameters.
    Parameters accept either scalar values (for 1 band or uniform values across bands)
    or lists of values per band.

    Parameters
    ----------
    item : pystac.Item
        The STAC item to add the asset to.
    name : str
        Asset key name (e.g., 'blue' for single-band, or filename stem for multiband).
    path : Path or str
        File path or URL to the asset.
    title : str, optional
        Human-readable title (e.g., 'Blue Band (B1)').
    description : str, optional
        Description of the asset.
    roles : list[str], optional
        Asset roles (e.g., ['data']).
    gsd : int or float, optional
        Ground sample distance.
    shape : list[int], optional
        Per-asset shape [height, width] for projection extension.
    transform : list, optional
        Per-asset affine transform for projection extension.
    band_names : str or list[str], optional
        Band name(s) (e.g., 'B1' or ['B02', 'B03']). Defaults to [name] if not provided.
    band_common_names : str or list[str], optional
        Common name(s) for each band (e.g., 'blue' or ['blue', 'green']).
    band_nodata : int, float or list, optional
        Nodata value(s) per band.
    band_dtypes : str or list[str], optional
        Data type string(s) per band (e.g., 'uint16').
    band_center_wavelengths : int, float or list, optional
        Center wavelength(s) per band.
    """
    if isinstance(path, Path):
        path = path.as_posix()

    # Build asset fields
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

    asset = pystac.Asset(**asset_fields)
    asset.set_owner(item)

    # Normalize band parameters to lists
    if band_names is None:
        b_names = [name]
    elif isinstance(band_names, str):
        b_names = [band_names]
    else:
        b_names = list(band_names)

    num_bands = len(b_names)
    b_common = _to_list(band_common_names, num_bands)
    b_nodata = _to_list(band_nodata, num_bands)
    b_dtypes = _to_list(band_dtypes, num_bands)
    b_wavelengths = _to_list(band_center_wavelengths, num_bands)

    # EO extension — band metadata
    eo = EOExtension.ext(asset, add_if_missing=True)
    eo_bands = []
    for bn, cn, cw in zip(b_names, b_common, b_wavelengths, strict=True):
        band_obj = Band.create(name=bn)
        if cn is not None:
            band_obj.common_name = cn
        if cw is not None:
            band_obj.center_wavelength = cw
        eo_bands.append(band_obj)
    eo.bands = eo_bands

    # Raster extension — per-band nodata and dtype
    if any(nd is not None for nd in b_nodata) or any(dt is not None for dt in b_dtypes):
        raster = RasterExtension.ext(asset, add_if_missing=True)
        raster_bands = []
        for nd, dt in zip(b_nodata, b_dtypes, strict=True):
            rb_kwargs: dict[str, Any] = {}
            if nd is not None:
                rb_kwargs["nodata"] = nd
            if dt is not None:
                rb_kwargs["data_type"] = dt
            raster_bands.append(RasterBand.create(**rb_kwargs))
        raster.bands = raster_bands

    # Projection extension — per-asset shape and transform
    if shape is not None or transform is not None:
        proj = ProjectionExtension.ext(asset, add_if_missing=True)
        if shape is not None:
            proj.shape = shape
        if transform is not None:
            proj.transform = transform

    item.add_asset(name, asset)

    if gsd is not None:
        item.assets[name].common_metadata.gsd = gsd


def filter_bands(dataset: pystac.Item, bands: list[str]) -> None:
    """Delete assets not in the given band list.

    Supports filtering by asset key names. For multiband assets, keeps an asset
    if its key or any of its eo:band names is in the bands list.
    """
    unneeded = [
        asset_key
        for asset_key in dataset.assets
        if asset_key not in bands and not _asset_has_band(dataset.assets[asset_key], bands)
    ]
    for asset_key in unneeded:
        dataset.assets.pop(asset_key)


def _asset_has_band(asset: pystac.Asset, bands: list[str]) -> bool:
    """Check if any eo:band name within an asset matches the bands list."""
    try:
        if asset.ext.has("eo") and asset.ext.eo.bands is not None:
            return any(b.name in bands for b in asset.ext.eo.bands)
    except Exception:  # noqa: S110
        pass
    return False


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


def _resolve_raster_var(asset: str, raster: Dataset) -> Optional[str]:
    """Find the raster data variable that corresponds to a STAC asset key.

    Handles both direct matches (asset key == data var name) and multiband
    naming convention where data vars are named "asset_key/band_name".

    Returns the matching data variable name, or None if not found.
    """
    if asset in raster:
        return asset
    # Check for multiband naming convention: "asset_key/band_name"
    for var in raster:
        var_str = str(var)
        if var_str.startswith(asset + "/"):
            return var_str
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

    mb = is_multi_asset(dataset)
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

    # Use file mtime if available, otherwise UTC now
    dataset.datetime = _get_time(output_path)

    # Adding hrefs
    for asset in dataset.assets:
        if not mb and (output_path.is_file() or len(output_path.suffixes) != 0) and ".json" not in output_path.suffixes:
            dataset.assets[asset].href = output_path.name
        else:
            dataset.assets[asset].href = dataset.assets[asset].ext.eo.bands[0].name + ".tif"

    # Changing datatypes and nodata
    for asset in dataset.assets:
        if dataset.assets[asset].ext.has("raster") and dataset.assets[asset].ext.raster.bands is not None:
            raster_bands = dataset.assets[asset].ext.raster.bands
            # For multiband assets with multiple raster:bands, update each
            if len(raster_bands) > 1 and dataset.assets[asset].ext.has("eo"):
                eo_bands = dataset.assets[asset].ext.eo.bands
                for i, rb in enumerate(raster_bands):
                    if i < len(eo_bands):
                        var_name = f"{asset}/{eo_bands[i].name}"
                        if var_name not in raster:
                            var_name = asset
                        if var_name in raster:
                            if rb.data_type is not None:
                                rb.data_type = raster[var_name].dtype.name
                            if rb.nodata is not None and raster[var_name].rio.nodata is not None:
                                rb.nodata = float(raster[var_name].rio.nodata)
            else:
                # Single-band asset
                if raster_bands[0].data_type is not None:
                    var_name = _resolve_raster_var(asset, raster)
                    if var_name is not None:
                        raster_bands[0].data_type = raster[var_name].dtype.name
                if raster_bands[0].nodata is not None:
                    var_name = _resolve_raster_var(asset, raster)
                    if var_name is not None and raster[var_name].rio.nodata is not None:
                        raster_bands[0].nodata = float(raster[var_name].rio.nodata)

    # Updating projection info — use first data var as reference
    ref_var = raster[next(iter(raster.data_vars))]
    dataset.ext.proj.shape = list(ref_var.shape[-2:])
    dataset.ext.proj.transform = list(ref_var.rio.transform())[:6]
    dataset.ext.proj.epsg = ref_var.rio.crs.to_epsg()
    # noinspection PyTypeChecker
    dataset.bbox = list(
        transform_bounds(
            ref_var.rio.crs,
            "EPSG:4326",
            *ref_var.rio.bounds(),
        ),
    )
    dataset.geometry = _bbox_to_polygon(dataset.bbox)

    # Updating projection info for each asset
    for asset in dataset.assets:
        resolved = _resolve_raster_var(asset, raster)
        if resolved is None:
            raise ValueError("Band " + str(asset) + " not found in raster.")

        # Changing projection info if resample is not None
        if dataset.assets[asset].ext.has("proj"):
            if dataset.assets[asset].ext.proj.shape is not None:
                dataset.assets[asset].ext.proj.shape = list(raster[resolved].shape[-2:])
            if dataset.assets[asset].ext.proj.transform is not None:
                dataset.assets[asset].ext.proj.transform = list(raster[resolved].rio.transform())[:6]

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
