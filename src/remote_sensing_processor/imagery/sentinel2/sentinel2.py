"""Sentinel-2 preprocessing."""

from pydantic import validate_call
from typing import List, Literal, Optional, Union

import warnings
import zipfile
from pathlib import Path
from random import randint

import dask
import xarray as xr

import rioxarray as rxr
import shapely

from pystac import Item

from remote_sensing_processor.common.common_functions import create_folder, delete, persist
from remote_sensing_processor.common.common_raster import (
    check_dtype,
    clip_to_initial_bbox,
    clipf,
    get_first_proj,
    get_initial_bbox,
    load_dataset,
    prepare_nodata,
    reproject,
    reproject_match,
    write,
)
from remote_sensing_processor.common.dataset import check_output
from remote_sensing_processor.common.fill import fillnodata
from remote_sensing_processor.common.torch_test import cuda_test
from remote_sensing_processor.common.types import (
    CRS,
    DirectoryPath,
    FilePath,
    ListOfPath,
    ListOfPystacItem,
    NewPath,
)
from remote_sensing_processor.imagery.sentinel2.dataset import (
    SENTINEL2_BANDS,
    postprocess_sentinel2_dataset,
    read_sentinel2_dataset,
)
from remote_sensing_processor.imagery.sentinel2.sen2cor import sen2correct
from remote_sensing_processor.imagery.sentinel2.superres import superresolution


@validate_call
def sentinel2(
    inputs: Union[ListOfPath, ListOfPystacItem],
    output_path: Optional[Union[DirectoryPath, NewPath]] = None,
    sen2cor: Optional[bool] = False,
    upscale: Optional[Literal["superres", "resample"]] = "superres",
    resample: Optional[str] = "bilinear",
    crs: Optional[Union[Literal["same"], CRS]] = None,
    cloud_mask: Optional[bool] = True,
    clip: Optional[FilePath] = None,
    normalize: Optional[bool] = False,
    write_stac: Optional[bool] = True,
) -> List[NewPath]:
    """
    Preprocess Sentinel-2 imagery.

    Parameters
    ----------
    inputs : string or list of strings or STAC Item or list of STAC Items
        Path to archive / directory or list of paths to archives / directories
        or a STAC Item or list of STAC Items (e.g., from Planetary Computer).
    output_path : string (optional)
        Path to output directory. If not set, then will write to the same directory as archive.
        Must be set if the inputs are remote STAC Items.
    sen2cor : bool (default = False)
        Is atmospheric correction using Sen2Cor needed. Set it to False if you have troubles with Sen2Cor.
        It will have no effect if Sentinel 2 product is already processed to L2A level.
        It cannot be applied to a remote STAC dataset.
        As Sen2Cor is an external product, we cannot guarantee that everything will work as expected.
        As L2A level products are available for most of the Sentinel-2 scenes,
        we strongly recommend using them instead of L1C level products
        to skip Sen2Cor stage and make processing faster and more stable.
    upscale : string or None (default = 'superres')
        Method for upscaling 20- and 60-m bands to 10 m resolution.
        Can be 'superres' - uses neural network for superresolution, 'resample' - uses resampling,
        or None - keeps original band resolution.
        Set it to 'resample' or None if you do not have GPU that supports CUDA.
    resample : resampling method from rasterio as a string (default = 'bilinear')
        Resampling method that will be used to upscale 20 and 60 m bands if upscale == 'resample'.
        You can read more about resampling methods
        `here <https://rasterio.readthedocs.io/en/latest/topics/resampling.html>`_.
    crs : any (optional)
        CRS in which output data should be or `same` to get CRS from the first archive.
    cloud_mask : bool (default = True)
        Is cloud masking needed.
    clip : string (optional)
        Path to a vector file to be used to crop the image.
    normalize : bool (default = False)
        Is min-max normalization to 0-1 range needed.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    list of pathlib.Paths
        List of paths where preprocessed Sentinel-2 products are saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> from glob import glob
        >>> sentinel2_imgs = glob("/home/rsp_test/sentinels/*.zip")
        >>> print(sentinel2_imgs)
        ['/home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626.zip',
         '/home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626.zip',
         '/home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626.zip',
         '/home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027.zip',
         '/home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624.zip',
         '/home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041.zip']
        >>> output_sentinels = rsp.sentinel2(sentinel2_imgs)
        Preprocessing of /home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624.zip completed
        Preprocessing of /home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041.zip completed
        >>> print(output_sentinels)
        ['/home/rsp_test/sentinels/S2A_MSIL2A_20210821T064626_N0209_R063_T42VWR_20210821T064626/S2A_MSIL2A_20210821T064626_N0209_R063_T42VWR_20210821T064626.json',
         '/home/rsp_test/sentinels/S2A_MSIL2A_20210821T064626_N0209_R063_T42WXS_20210821T064626/S2A_MSIL2A_20210821T064626_N0209_R063_T42WXS_20210821T064626.json',
         '/home/rsp_test/sentinels/S2B_MSIL2A_20210821T064626_N0209_R020_T43VCL_20210821T064626/S2B_MSIL2A_20210821T064626_N0209_R020_T43VCL_20210821T064626.json',
         '/home/rsp_test/sentinels/S2A_MSIL2A_20210626T063027_N0209_R120_T43VDK_20210626T063027/S2A_MSIL2A_20210626T063027_N0209_R120_T43VDK_20210626T063027.json',
         '/home/rsp_test/sentinels/S2A_MSIL2A_20210823T063624_N0209_R120_T43VDL_20210823T063624/S2A_MSIL2A_20210823T063624_N0209_R120_T43VDL_20210823T063624.json',
         '/home/rsp_test/sentinels/S2B_MSIL2A_20210709T064041_N0209_R020_T43VDL_20210709T064041/S2B_MSIL2A_20210709T064041_N0209_R020_T43VDL_20210709T064041.json']
    """
    # TODO: Update Sentinel-2 processing when migration to Zarr will be completed
    # TODO: Remove Sen2Cor
    # TODO: Add more upscaling methods, maybe from https://github.com/matciotola/Sentinel2-SR-Toolbox
    if cuda_test() is False and upscale == "superres":
        warnings.warn("CUDA or MPS is not available. Superresolution process could be very slow.", stacklevel=1)

    paths = []
    for path in inputs:
        if crs == "same":
            crs = get_first_proj(path)
        outfile = sentinel2_proc(
            path=path,
            output_path=output_path,
            sen2cor=sen2cor,
            upscale=upscale,
            resample=resample,
            crs=crs,
            cloud_mask=cloud_mask,
            clip=clip,
            normalize=normalize,
            write_stac=write_stac,
        )
        paths.append(outfile)
        print("Preprocessing of " + str(path) + " completed")
    return paths


def sentinel2_proc(
    path: Union[Path, Item],
    output_path: Optional[Path] = None,
    sen2cor: bool = False,
    upscale: Optional[Literal["superres", "resample"]] = "superres",
    resample: Optional[str] = "bilinear",
    crs: Optional[CRS] = None,
    cloud_mask: Optional[bool] = True,
    clip: Optional[Path] = None,
    normalize: Optional[bool] = False,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """Process a single Sentilel-2 product."""
    output_path = check_output(path, output_path, parent=True)

    # Loading dataset
    dataset, mtd = read_sentinel2_dataset(path)
    if dataset.common_metadata.description not in ["Sentinel2_up_l1", "Sentinel2_up_l2"]:
        raise ValueError("Cannot read Sentinel-2 product")
    if mtd.mtd is None:
        raise ValueError("Cannot read Sentinel-2 MTD")

    # If Sen2Cor correction is needed and the product is of level 1
    if dataset.common_metadata.description == "Sentinel2_up_l1" and sen2cor is True:
        warnings.warn(
            "As almost all of Sentinel 2 data is now available at level 2, sen2cor will be deprecated at RSP 0.4",
            DeprecationWarning,
            stacklevel=2,
        )
        if mtd.root is None:
            raise ValueError("Can not apply sen2cor to a remote STAC dataset")
        # Creating a temporary directory
        temp_dir = path.parent / ("temp_" + str(randint(1000, 9999)))
        create_folder(temp_dir)
        # Moving sentinel data to a temporary directory
        copy_sentinel(mtd.root, temp_dir)
        # Getting a path to l1 product in a temp dir
        workpath = next(iter(temp_dir.glob("*/")))
        # Running Sen2Cor
        sen2correct(workpath)
        # Loading processed dataset
        dataset, mtd = read_sentinel2_dataset(temp_dir)
        if dataset.common_metadata.description not in ["Sentinel2_up_l1", "Sentinel2_up_l2"]:
            raise ValueError("Cannot read Sentinel-2 product")
        if mtd.mtd is None:
            raise ValueError("Cannot read Sentinel-2 MTD")
    else:
        temp_dir = None

    # Creating an output directory or cleaning the directory if already exists
    create_folder(output_path / dataset.id)

    # Reading datasets with different gsd
    futures = []
    data60, bbox = read_sentinel2_gsd(dataset, clip, 60)
    data60 = persist(data60)
    for gsd in [10, 20]:
        futures.append(dask.delayed(read_sentinel2_gsd)(dataset, clip, gsd, bbox))
    data10, data20 = dask.compute(*futures)
    data10, data20 = persist(data10, data20)

    # Upscaling
    if upscale is not None:
        data20 = reproject_match(data20, data10, resample)
        data60 = reproject_match(data60, data10, resample)
        if upscale == "superres":
            img = superresolution(data10=data10, data20=data20, data60=data60, dataset=dataset)
        elif upscale == "resample":
            img = xr.merge([data10, data20, data60])
        else:
            raise ValueError("Unknown resampling type")
    else:
        img = [data10, data20, data60]

    outimgs = []
    if isinstance(img, list):
        for i in img:
            outimgs.append(
                s2postprocess(
                    img=i,
                    dataset=dataset,
                    crs=crs,
                    cloud_mask=cloud_mask,
                    clip=clip,
                    normalize=normalize,
                    path=output_path,
                ),
            )
    else:
        outimgs.append(
            s2postprocess(
                img=img,
                dataset=dataset,
                crs=crs,
                cloud_mask=cloud_mask,
                clip=clip,
                normalize=normalize,
                path=output_path,
            ),
        )

    # Removing temp dir
    if temp_dir is not None:
        delete(temp_dir)

    json_path = output_path / dataset.id / (dataset.id + ".json")
    # Updating STAC dataset
    postprocess_sentinel2_dataset(dataset, outimgs, json_path, upscale)
    if write_stac:
        # Writing JSON metadata file
        dataset.save_object(dest_href=json_path)
        return json_path
    return json_path.parent


def copy_sentinel(source: Path, dest: Path) -> None:
    """Copies sentinel data to a temp directory for sen2cor processing."""
    if ".zip" in str(source):
        for parent in source.parents:
            if ".zip" in parent.suffixes:
                with zipfile.ZipFile(parent) as archive:
                    archive.extractall(dest)


def read_sentinel2_gsd(
    dataset: Item,
    clip: Optional[Path],
    gsd: int,
    bbox: Optional[list] = None,
) -> Union[xr.Dataset, tuple[xr.Dataset, Optional[list]]]:
    """This function reads Sentinel 2 subdataset with specific gsd."""
    # Filtering bands
    bands = [
        asset
        for asset in dataset.assets
        if (asset in SENTINEL2_BANDS) and (dataset.assets[asset].common_metadata.gsd == gsd)
    ]

    # Reading bands
    img = load_dataset(dataset, bands, clip=None)

    # Reading only the area we actually need
    if clip is not None and bbox is None:
        bbox = get_initial_bbox(img, clip)
        img = clip_to_initial_bbox(img, bbox)
        bbox = [shapely.geometry.box(*list(img.rio.bounds()))]
        return img, bbox
    if clip is not None and bbox is not None:
        return clip_to_initial_bbox(img, bbox)
    if gsd == 60:
        return img, None
    return img


def s2postprocess(
    img: xr.Dataset,
    dataset: Item,
    crs: Optional[CRS],
    cloud_mask: Optional[bool],
    clip: Optional[Path],
    normalize: Optional[bool],
    path: Path,
) -> xr.Dataset:
    """Sentinel-2 post processing."""
    img, nodata = prepare_nodata(img, default_nodata=0)

    # Cloud masking
    if cloud_mask:
        img = mask_sentinel2(img, dataset, nodata)

    # Reprojection
    if crs is not None:
        img = reproject(img, crs)

    # Clipping
    if clip is not None:
        img = clipf(img, clip)

    # Normalization
    if normalize:
        with xr.set_options(keep_attrs=True):
            img = img / 10000

    img = check_dtype(img)

    # Save
    results = []
    outfiles = []
    for band in img:
        pathres = path / dataset.id / (dataset.assets[band].ext.eo.bands[0].name + ".tif")
        outfiles.append(pathres)
        results.append(write(img[band], pathres, compute=False))
    dask.compute(*results)
    return img


def mask_sentinel2(img: xr.Dataset, dataset: Item, nodata: Optional[Union[int, float]]) -> xr.Dataset:
    """Masking clouds and incorrect data."""
    # Masking clouds by mask cloud mask
    if "cldmsk" in dataset.assets:
        shape = dataset.assets["cldmsk"].href
        if shape.endswith(".gml"):
            img = clipf(img, Path(shape), invert=True)
            mask = None
        else:
            with rxr.open_rasterio(shape, chunks=True, lock=True) as tif:
                mask = persist(tif.squeeze())
            mask = reproject_match(mask, img)
            mask = xr.where((mask[0] == 0) & (mask[1] == 0), 1, 0)  # 0 - nodata, 1 - data
    else:
        mask = None

    # Masking clouds by scl mask
    if "scl" in dataset.assets:
        with rxr.open_rasterio(dataset.assets["scl"].href, chunks=True, lock=True) as tif:
            sclmsk = persist(tif.squeeze())
        sclmsk = reproject_match(sclmsk, img)
        if mask is not None:
            mask = xr.where(sclmsk.isin([0, 1, 3, 8, 9, 10]), 0, mask.data)  # 0 - nodata, 1 - data
        else:
            mask = xr.where(sclmsk.isin([0, 1, 3, 8, 9, 10]), 0, 1)  # 0 - nodata, 1 - data

    # Masking clouds by cloud proba
    if "cldprb" in dataset.assets:
        with rxr.open_rasterio(dataset.assets["cldprb"].href, chunks=True, lock=True) as tif:
            clmask = persist(tif.squeeze())
        clmask = reproject_match(clmask, img)
        if mask is not None:  # noqa: SIM108
            mask = mask.where(clmask.data < 10, 0)  # 0 - nodata, 1 - data
        else:
            mask = xr.where(clmask.data >= 10, 0, 1)  # 0 - nodata, 1 - data

    if mask is not None:
        # Delete small gaps in mask
        mask = xr.where(fillnodata(mask, mask, 5, None) == 1, 0, 1)  # fill 0 - nodata -> 1 - nodata, 0 - data
        # Emerge mask
        mask = fillnodata(mask, mask, 22, None)  # fill 0 - data
        # Masking nodata
        img = img.where(mask.data == 0, nodata)  # 0 - data, 1 - nodata
        img = persist(img)
    return img
