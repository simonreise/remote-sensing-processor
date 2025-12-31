"""Landsat preprocessing."""

from pydantic import validate_call
from typing import Any, Literal, Optional, Union

import warnings
from pathlib import Path
from urllib.parse import urlparse

import fsspec

import dask
from xarray import DataArray, Dataset

import satpy

from pystac import Item

from remote_sensing_processor.common.common_functions import create_folder, persist
from remote_sensing_processor.common.common_raster import (
    check_dtype,
    clip_to_initial_bbox,
    clipf,
    get_first_proj,
    get_initial_bbox,
    prepare_nodata,
    reproject,
    reproject_match,
    restore_nodata_from_nan,
    unpack_bitmap,
    write,
)
from remote_sensing_processor.common.dataset import check_output
from remote_sensing_processor.common.types import (
    CRS,
    DirectoryPath,
    FilePath,
    ListOfPath,
    ListOfPystacItem,
    NewPath,
    Temperature,
)
from remote_sensing_processor.imagery.landsat.dataset import (
    postprocess_landsat_dataset,
    read_landsat_dataset,
)
from remote_sensing_processor.imagery.types import get_type


warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=".*You will likely lose important projection information.*")


@validate_call
def landsat(
    inputs: Union[ListOfPath, ListOfPystacItem],
    output_path: Optional[Union[DirectoryPath, NewPath]] = None,
    crs: Optional[Union[Literal["same"], CRS]] = None,
    cloud_mask: Optional[bool] = True,
    pansharpen: Optional[bool] = True,
    keep_pan_band: Optional[bool] = False,
    resample: Optional[str] = "gradient_search",
    clip: Optional[FilePath] = None,
    t: Optional[Temperature] = "k",
    normalize: Optional[bool] = False,
    write_stac: Optional[bool] = True,
) -> list[NewPath]:
    """
    Preprocess Landsat imagery.

    Parameters
    ----------
    inputs : string or list of strings or STAC Item or list of STAC Items
        Path to archive / directory or list of paths to archives / directories
        or a STAC Item or list of STAC Items (e.g., from Planetary Computer).
    output_path : string (optional)
        Path to output directory. If not set, then will write to the same directory as archive.
        Must be set if the inputs are remote STAC Items.
    crs : any (optional)
        CRS in which output data should be or `same` to get CRS from the first archive.
    cloud_mask : bool (default = True)
        Is cloud masking needed.
    pansharpen : bool (default = True)
        Is pansharpening needed. RSP uses Brovey transform for pansharpening Landsat 7, 8 and 9.
    keep_pan_band : bool (default = False)
        Keep a pansharpening band or delete it. The pansharpening band has the same wavelengths as optical bands,
        so it does not contain any additional information to other bands. Affects only Landsat 7, 8 and 9.
    resample : resampling method from satpy as a string (default = 'gradient_search')
        Resampling method that will be used to upscale bands that cannot be upscaled in pansharpening operation.
        You can read more about resampling methods
        `here <https://satpy.readthedocs.io/en/stable/api/satpy.resample.html#resampling-algorithms>`_.
        Affects only Landsat 7, 8 and 9.
    clip : string (optional)
        Path to a vector file to be used to crop the image.
    t : string ('k' or 'c', default = 'k')
        Convert thermal bands to kelvins or celsius (no fahrenheit lol).
    normalize : bool (default = False)
        Is min-max normalization to 0-1 range needed.
    write_stac : bool (default = True)
        If True, then output metadata is saved to a STAC file.

    Returns
    -------
    list of pathlib.Paths
        List of paths where preprocessed Landsat products are saved.

    Examples
    --------
        >>> import remote_sensing_processor as rsp
        >>> from glob import glob
        >>> landsat_imgs = glob("/home/rsp_test/landsat/*.tar")
        >>> print(landsat_imgs)
        ['/home/rsp_test/landsat/LC08_L1TP_160023_20210825_20210901_02_T1.tar',
         '/home/rsp_test/landsat/LT05_L1TP_160023_20110814_20200820_02_T1.tar',
         '/home/rsp_test/landsat/LE07_L1TP_159023_20210826_20210921_02_T1.tar',
         '/home/rsp_test/landsat/LT05_L1TP_162023_20110812_20200820_02_T1.tar',
         '/home/rsp_test/landsat/LM05_L1TP_161023_19930803_20211018_02_T2.tar']
        >>> output_landsats = rsp.landsat(landsat_imgs)
        Preprocessing of /home/rsp_test/landsat/LC08_L1TP_160023_20210825_20210901_02_T1.tar completed
        Preprocessing of /home/rsp_test/landsat/LT05_L1TP_160023_20110814_20200820_02_T1.tar completed
        Preprocessing of /home/rsp_test/landsat/LE07_L1TP_159023_20210826_20210921_02_T1.tar completed
        Preprocessing of /home/rsp_test/landsat/LT05_L1TP_162023_20110812_20200820_02_T1.tar completed
        Preprocessing of /home/rsp_test/landsat/LM05_L1TP_161023_19930803_20211018_02_T2.tar completed
        >>> print(output_landsats)
        ['/home/rsp_test/landsat/LC08_L1TP_160023_20210825_20210901_02_T1/LC08_L1TP_160023_20210825_20210901_02_T1.json',
         '/home/rsp_test/landsat/LT05_L1TP_160023_20110814_20200820_02_T1/LT05_L1TP_160023_20110814_20200820_02_T1.json',
         '/home/rsp_test/landsat/LE07_L1TP_159023_20210826_20210921_02_T1/LE07_L1TP_159023_20210826_20210921_02_T1.json',
         '/home/rsp_test/landsat/LT05_L1TP_162023_20110812_20200820_02_T1/LT05_L1TP_162023_20110812_20200820_02_T1.json',
         '/home/rsp_test/landsat/LM05_L1TP_161023_19930803_20211018_02_T2/LM05_L1TP_161023_19930803_20211018_02_T2.json']
    """
    paths = []
    for path in inputs:
        if crs == "same":
            crs = get_first_proj(path)
        outfile = landsat_proc(
            path=path,
            output_path=output_path,
            crs=crs,
            cloud_mask=cloud_mask,
            pansharpen=pansharpen,
            keep_pan_band=keep_pan_band,
            resample=resample,
            t=t,
            clip=clip,
            normalize=normalize,
            write_stac=write_stac,
        )
        paths.append(outfile)
        print("Preprocessing of " + str(path) + " completed")
    return paths


def landsat_proc(
    path: Union[Path, Item],
    output_path: Optional[Union[DirectoryPath, NewPath]] = None,
    crs: Optional[CRS] = None,
    cloud_mask: Optional[bool] = True,
    pansharpen: Optional[bool] = True,
    keep_pan_band: Optional[bool] = False,
    resample: Optional[str] = "gradient_search",
    t: Optional[Temperature] = "k",
    clip: Optional[FilePath] = None,
    normalize: Optional[bool] = False,
    write_stac: Optional[bool] = True,
) -> NewPath:
    """Process a single Landsat product."""
    output_path = check_output(path, output_path, parent=True)

    rsp_type = get_type(path)

    files = get_files(path, rsp_type)
    scene = satpy.Scene(filenames=files)

    # Get band names
    spectral_bands, thermal_bands, pan_band, qa_bands = get_bands(files)
    if pan_band is not None and pansharpen:
        all_bands = spectral_bands + thermal_bands + pan_band
    else:
        all_bands = spectral_bands + thermal_bands
    if cloud_mask:
        all_bands += qa_bands
    ok_bands = [band for band in all_bands if band in scene.available_dataset_names()]

    scene.load(ok_bands)

    dataset = read_landsat_dataset(path, scene, rsp_type)

    # Clipping data to bbox
    bbox = get_initial_bbox(scene.to_xarray_dataset(), clip)
    if bbox is not None:
        scene = scene.crop(xy_bbox=bbox[0].bounds)
    scene = persist(scene)

    scene = scene.resample(resampler=resample)
    scene = persist(scene)

    img = scene.to_xarray_dataset()

    # Rename bands according to STAC convention
    img = img.rename(
        {
            band: asset
            for asset in dataset.assets
            for band in ok_bands
            if band == dataset.assets[asset].ext.eo.bands[0].name
        },
    )

    # Set up nodata value
    img, _ = prepare_nodata(img, default_nodata=0)

    # Normalizing and processing temperature bands
    img = img.map(process, t=t, normalize=normalize)
    img = persist(img)

    # Pansharpening
    pan = None
    if pan_band is None:
        pansharpen = False
        keep_pan_band = False
    if (pansharpen is True) or (keep_pan_band is True):
        if pansharpen:
            pan_c = get_pan_coef(img, dataset)
            img = img.map(pansharp, pan_c=pan_c.data, rsp_type=rsp_type, normalize=normalize)
            if not keep_pan_band:
                img = img.drop_vars(["pan"])
            img = persist(img)
        else:
            scene = satpy.Scene(filenames=files)
            scene.load(pan_band)
            pan = scene["B8"]
            pan = clip_to_initial_bbox(pan, bbox)
            pan = persist(pan)
            pan, _ = prepare_nodata(pan, default_nodata=0)
            pan = process(pan, t=t, normalize=normalize)
            pan = persist(pan)

    # Reading quality assessment band and masking clouds
    if cloud_mask:
        qa = img[qa_bands]
        img = img.drop_vars(qa_bands)
        img, pan = get_mask(img, qa, dataset, pan)

    # Reprojecting
    if crs is not None:
        img = reproject(img, crs)
        if (pansharpen is False) and (keep_pan_band is True):
            pan = reproject(pan, crs)

    # Clipping
    if clip is not None:
        img = clipf(img, clip)
        if (pansharpen is False) and (keep_pan_band is True):
            pan = clipf(pan, clip)

    img = check_dtype(img)
    if (pansharpen is False) and (keep_pan_band is True):
        pan = check_dtype(pan)

    img = restore_nodata_from_nan(img)
    if (pansharpen is False) and (keep_pan_band is True):
        pan = restore_nodata_from_nan(pan)

    # Save
    outfiles = []
    results = []
    # Creating an output folder
    json_path = output_path / dataset.id / (dataset.id + ".json")
    # Updating STAC dataset
    postprocess_landsat_dataset(dataset, img, json_path, pansharpen, keep_pan_band, pan)
    # Creating an output directory or cleaning the directory if already exists
    create_folder(output_path / dataset.id)
    # Writing files
    for band in img:
        pathres = output_path / dataset.id / dataset.assets[band].href
        outfiles.append(pathres)
        results.append(write(img[band], pathres, compute=False))
    if (pansharpen is False) and (keep_pan_band is True):
        pathres = output_path / dataset.id / dataset.assets["pan"].href
        outfiles.append(pathres)
        results.append(write(pan, pathres, compute=False))
    dask.compute(*results)

    if write_stac:
        # Writing JSON metadata file
        dataset.save_object(dest_href=json_path.as_posix())
        return json_path
    return json_path.parent


def get_files(path: Union[Path, Item], rsp_type: str) -> dict[str, dict[str, Any]]:
    """Get files from the input path."""
    # Construct reader name
    if "olitirs" in rsp_type:
        sensor = "oli_tirs"
    elif "etm" in rsp_type:
        sensor = "etm"
    elif "tm" in rsp_type:
        sensor = "tm"
    elif "mss" in rsp_type:
        sensor = "mss"
    else:
        raise ValueError("Unknown Landsat product")
    if "l1" in rsp_type:
        level = "l1"
    elif "l2" in rsp_type:
        level = "l2"
    else:
        raise ValueError("Unknown Landsat product")
    reader = sensor + "_" + level + "_tif"

    if isinstance(path, Item):
        urls = [path.assets[i].href for i in path.assets]
        reference = {
            "version": 1,
            "templates": {},
            "refs": {Path(urlparse(url).path).name: [url] for url in urls},
        }
        fs = fsspec.filesystem("reference", fo=reference)
        files = satpy.find_files_and_readers(base_dir="", reader=reader, fs=fs)
        for key in files:
            files[key] = [satpy.readers.core.remote.FSFile(file, fs=fs) for file in files[key]]
    elif path.is_file() and (
        (".tar" in path.suffixes) or (".gz" in path.suffixes) or (".TAR" in path.suffixes) or (".GZ" in path.suffixes)
    ):
        fs = fsspec.filesystem("tar", fo=path.as_posix())
        files = satpy.find_files_and_readers(base_dir="", reader=reader, fs=fs)
        for key in files:
            files[key] = [satpy.readers.core.remote.FSFile(file, fs=fs) for file in files[key]]
    elif path.is_dir():
        files = satpy.find_files_and_readers(base_dir=path.as_posix(), reader=reader)
        for key in files:
            files[key] = [Path(file).resolve() for file in files[key]]
    else:
        raise ValueError(str(path) + " is not a directory, tar/tar.gz archive or STAC object.")
    return files


def get_bands(files: dict[str, dict[str, Any]]) -> tuple[list[str], list[str], Optional[list[str]], list[str]]:
    """Get band names for a specific Landsat product."""
    if "oli_tirs_l1_tif" in files:
        spectral_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9"]
        thermal_bands = ["B10", "B11"]
    elif "oli_tirs_l2_tif" in files:
        spectral_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        thermal_bands = ["B10"]
    elif "etm_l1_tif" in files:
        spectral_bands = ["B1", "B2", "B3", "B4", "B5", "B7"]
        thermal_bands = ["B6_VCID_1", "B6_VCID_2"]
    elif "etm_l2_tif" in files or "tm_l1_tif" in files or "tm_l2_tif" in files:
        spectral_bands = ["B1", "B2", "B3", "B4", "B5", "B7"]
        thermal_bands = ["B6"]
    else:
        spectral_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        thermal_bands = []
    pan_band = ["B8"] if "oli_tirs_l1_tif" in files or "etm_l1_tif" in files else None
    if "oli_tirs_l2_tif" in files:
        qa_bands = ["qa", "qa_radsat", "qa_aerosol", "qa_st"]
    elif "etm_l2_tif" in files or "tm_l2_tif" in files:
        qa_bands = ["qa", "qa_radsat", "qa_atmos_opacity", "qa_cloud", "qa_st"]
    else:
        qa_bands = ["qa", "qa_radsat"]
    return spectral_bands, thermal_bands, pan_band, qa_bands


def process(img: DataArray, t: Optional[Temperature] = "k", normalize: Optional[bool] = False) -> DataArray:
    """Normalize and modify temperature bands."""
    if img.name in ["lwir11", "lwir12"]:
        # Change temperature units
        if t == "c":
            img = img - 273.15
            deg = 273.15
        elif t == "k":
            deg = 0
        else:
            raise ValueError("wtf is a fahrenheit")
        # Normalize temperature in range 175 k - 375 k
        if normalize:
            img = (img - (175 - deg)) / ((375 - deg) - (175 - deg))
    elif normalize:
        img = img * 0.01
    return img.clip(0, None)


def get_mask(
    img: Dataset,
    qa: Dataset,
    dataset: Item,
    pan: Optional[DataArray] = None,
) -> tuple[Dataset, Optional[DataArray]]:
    """Gets a mask from Quality assessment bands."""
    # QA Pixel band
    qa_pixel = qa["qa"].astype("int16")
    qa_pixel = unpack_bitmap(qa_pixel)
    qa_pixel = (
        (qa_pixel[0] == 1)  # Fill
        | (qa_pixel[1] == 1)  # Dilated cloud
        | (qa_pixel[2] == 1)  # Cirrus
        | (qa_pixel[3] == 1)  # Cloud
        | (qa_pixel[4] == 1)  # Cloud shadow
        | ((qa_pixel[8] == 0) & (qa_pixel[9] == 1))  # Medium cloud confidence
        | ((qa_pixel[8] == 1) & (qa_pixel[9] == 1))  # High cloud confidence
        | ((qa_pixel[10] == 1) & (qa_pixel[11] == 1))  # High cloud shadow confidence
        | ((qa_pixel[14] == 1) & (qa_pixel[15] == 1))  # High cirrus confidence
    )
    img = img.where(qa_pixel.data == 0)
    if pan is not None:
        qa_pixel = reproject_match(qa_pixel, pan)
        pan = pan.where(qa_pixel.data == 0)

    # QA Radsat band
    if "qa_radsat" in img:
        qar = qa["qa_radsat"].astype("int16")
        qar = unpack_bitmap(qar)

        def mask_qar(data: DataArray) -> DataArray:
            """Mask saturarted areas in a specific band."""
            btitle = dataset.assets[str(data.name)].ext.eo.bands[0].name[1:]
            if btitle.isdigit() and int(btitle) < 9:
                btitle = int(btitle) - 1
            elif btitle == "6_VCID_1":
                btitle = 5
            elif btitle == "6_VCID_2":
                btitle = 8
            if isinstance(btitle, int):
                data = data.where(qar[btitle].data == 0)
            return data

        img = img.map(mask_qar)
        img = img.where(qar[9].data == 0)
        if pan is not None:
            qar = reproject_match(qar, pan)
            pan = pan.where(qar[9].data == 0)

    # QA Aerosol
    if "qa_aerosol" in img:
        qa_aerosol = qa["qa_aerosol"].astype("int8")
        qa_aerosol = unpack_bitmap(qa_aerosol)
        qa_aerosol = (qa_aerosol[6] == 1) and (qa_aerosol[7] == 1)  # High aerosol level
        img = img.where(qa_aerosol.data == 0)
        if pan is not None:
            qa_aerosol = reproject_match(qa_aerosol, pan)
            pan = pan.where(qa_aerosol.data == 0)

    # QA Cloud
    if "qa_cloud" in img:
        qa_cloud = qa["qa_cloud"].astype("int8")
        qa_cloud = unpack_bitmap(qa_cloud)
        qa_cloud = (
            (qa_cloud[1] == 1)  # Cloud
            | (qa_cloud[2] == 1)  # Cloud shadow
            | (qa_cloud[3] == 1)  # Adjacent to cloud
        )
        img = img.where(qa_cloud.data == 0)
        if pan:
            qa_cloud = reproject_match(qa_cloud, pan)
            pan = pan.where(qa_cloud.data == 0)

    # QA Atmospheric opacity
    if "qa_atmos_opacity" in img:
        qa_ao = qa["qa_atmos_opacity"]
        img = img.where(qa_ao.data > 0.3)
        if pan is not None:
            qa_ao = reproject_match(qa_ao, pan)
            pan = pan.where(qa_ao.data > 0.3)

    # QA ST
    if "qa_st" in img:
        qa_st = qa["qa_st"]
        img = img.where(qa_st.data > 15)
        if pan is not None:
            qa_st = reproject_match(qa_st, pan)
            pan = pan.where(qa_st.data > 15)

    img = persist(img)
    if pan is not None:
        pan = persist(pan)
    # TODO: this is a fix for an odc.reproject error that replaces 0 with -1 if nodata not set, remove it when fixed
    # mask.rio.write_nodata(-1, inplace=True)
    return img, pan


def get_pan_coef(img: Dataset, dataset: Item) -> DataArray:
    """Calculates panchromatic coefficient."""
    if dataset.common_metadata.description == "Landsat_olitirs_up_l1":
        return (img.pan / ((0.42 * img.blue) + (0.98 * img.green) + (0.6 * img.red)) / 2).squeeze()
    if dataset.common_metadata.description == "Landsat_etm_up_l1":
        return (img.pan / ((0.42 * img.blue) + (0.98 * img.green) + (0.6 * img.red) + (1 * img.nir08)) / 3).squeeze()
    raise ValueError(f"{dataset.common_metadata.description} cannot be processed")


def pansharp(img: DataArray, pan_c: DataArray, rsp_type: str, normalize: Optional[bool] = False) -> DataArray:
    """Performs pansharpening of a single band."""
    if (rsp_type == "Landsat_olitirs_up_l1" and img.name in ["blue", "green", "red"]) or (
        rsp_type == "Landsat_etm_up_l1" and img.name in ["blue", "green", "red", "nir"]
    ):
        if normalize:
            return (img * pan_c).clip(0, 1)
        return (img * pan_c).clip(0, 100)
    return img
