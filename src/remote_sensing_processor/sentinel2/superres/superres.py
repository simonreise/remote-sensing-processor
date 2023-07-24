import sys
import os
import gc
import re
#import json
from collections import defaultdict
#import subprocess

from typing import List, Tuple
from pathlib import Path
import glob
import warnings

import numpy as np
import rasterio
#from geojson import FeatureCollection
from rasterio.windows import Window
from rasterio import Affine as A
import pyproj as proj

from remote_sensing_processor.sentinel2.superres.supres import dsen2_20, dsen2_60

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# This code is adapted from this repository
# https://github.com/lanha/DSen2 and is distributed under the same
# license.

class Superresolution():
    def __init__(
        self,
        geometry = None,
        bounds = None,
        copy_original_bands = False,
        clip_to_aoi = False,
        output_dir: str = "/tmp/output/",
        input_dir: str = "/tmp/input/",
        data_folder: str = "MTD*.xml"):
        self.geometry = geometry
        self.bounds = bounds
        self.copy_original_bands = copy_original_bands
        self.clip_to_aoi = clip_to_aoi
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.data_folder = data_folder
    
    @staticmethod
    def check_size(dims):
        xmin, ymin, xmax, ymax = dims
        if xmax < xmin or ymax < ymin:
            raise ValueError(                
                "Invalid region of interest / UTM Zone combination",
            )

        if (xmax - xmin) < 192 or (ymax - ymin) < 192:
            raise ValueError(
                "AOI too small. Try again with a larger AOI (minimum pixel width or heigh of 192)",
            )
            
    #Now useless
    """def get_final_json(self) -> FeatureCollection:
        
        #This method return an output json file.
        
        input_metadata = load_metadata()
        feature_list = []
        for feature in input_metadata.features:
            path_to_input_img = feature["properties"]["up42.data_path"]
            path_to_output_img = Path(path_to_input_img).stem + "_superresolution.tif"
            out_feature = feature.copy()
            if self.params.__dict__["clip_to_aoi"]:
                out_feature["geometry"] = self.params.geometry()
                out_feature["bbox"] = self.bounds()
            out_feature["properties"]["up42.data_path"] = path_to_output_img
            feature_list.append(out_feature)
        out_fc = FeatureCollection(feature_list)

        return out_fc"""
        
    
    def get_data(self) -> Tuple[List, str]:
        """
        This method returns the raster data set of original image for
        all the available resolutions.
        """
        data_path = ""
        for file in glob.iglob(
            os.path.join(self.input_dir, self.data_folder),
            recursive=True,
        ):
            data_path = file

        # The following line will define whether image is L1C or L2A
        # For instance image_level can be "MSIL1C" or "MSIL2A"
        image_level = Path(data_path).stem.split("_")[1]
        raster_data = rasterio.open(data_path)
        datasets = raster_data.subdatasets

        return datasets, image_level

    @staticmethod
    def get_max_min(x_1: int, y_1: int, x_2: int, y_2: int, data) -> Tuple:
        """
        This method gets pixels' location for the region of interest on the 10m bands
        and returns the min/max in each direction and to nearby 60m pixel boundaries
        and the area associated to the region of interest.

        Examples:
            >>> get_max_min(0,0,400,400)
            (0, 0, 395, 395, 156816)

        """
        with rasterio.open(data) as d_s:
            d_width = d_s.width
            d_height = d_s.height

        tmxmin = max(min(x_1, x_2, d_width - 1), 0)
        tmxmax = min(max(x_1, x_2, 0), d_width - 1)
        tmymin = max(min(y_1, y_2, d_height - 1), 0)
        tmymax = min(max(y_1, y_2, 0), d_height - 1)
        # enlarge to the nearest 60 pixel boundary for the super-resolution
        tmxmin = int(tmxmin / 6) * 6
        tmxmax = int((tmxmax + 1) / 6) * 6 - 1
        tmymin = int(tmymin / 6) * 6
        tmymax = int((tmymax + 1) / 6) * 6 - 1
        area = (tmxmax - tmxmin + 1) * (tmymax - tmymin + 1)
        return tmxmin, tmymin, tmxmax, tmymax, area

    # pylint: disable-msg=too-many-locals
    def to_xy(self, lon: float, lat: float, data) -> Tuple:
        """
        This method gets the longitude and the latitude of a given point and projects it
        into pixel location in the new coordinate system.

        Args:
            lon: The longitude of a chosen point
            lat: The longitude of a chosen point

        Returns:
            The pixel location in the coordinate system of the input image
        """
        # get the image's coordinate system.
        with rasterio.open(data) as d_s:
            coor = d_s.transform
        a_t, b_t, xoff, d_t, e_t, yoff = [coor[x] for x in range(6)]

        # transform the lat and lon into x and y position which are defined in
        # the world's coordinate system.
        local_crs = self.get_utm(data)
        crs_wgs = proj.Proj(init="epsg:4326")  # WGS 84 geographic coordinate system
        crs_bng = proj.Proj(init=local_crs)  # use a locally appropriate projected CRS
        x_p, y_p = proj.transform(crs_wgs, crs_bng, lon, lat)
        x_p -= xoff
        y_p -= yoff

        # matrix inversion
        # get the x and y position in image's coordinate system.
        det_inv = 1.0 / (a_t * e_t - d_t * b_t)
        x_n = (e_t * x_p - b_t * y_p) * det_inv
        y_n = (-d_t * x_p + a_t * y_p) * det_inv
        return int(x_n), int(y_n)

    @staticmethod
    def get_utm(data) -> str:
        """
        This method returns the utm of the input image.

        Args:
            data: The raster file for a specific resolution.

        Returns:
            UTM of the selected raster file.
        """
        with rasterio.open(data) as d_s:
            data_crs = d_s.crs.to_dict()
        utm = data_crs["init"]
        return utm

    # pylint: disable-msg=too-many-locals
    def area_of_interest(self, data):
        """
        This method returns the coordinates that define the desired area of interest.
        """

        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = self.bounds()
        x_1, y_1 = self.to_xy(roi_lon1, roi_lat1, data)
        x_2, y_2 = self.to_xy(roi_lon2, roi_lat2, data)
        xmi, ymi, xma, yma, area = self.get_max_min(x_1, y_1, x_2, y_2, data)
        return xmi, ymi, xma, yma, area

    @staticmethod
    def validate_description(description: str) -> str:
        """
        This method rewrites the description of each band in the given data set.

        Args:
            description: The actual description of a chosen band.

        Examples:
            >>> ds10.descriptions[0]
            'B4, central wavelength 665 nm'
            >>> validate_description(ds10.descriptions[0])
            'B4 (665 nm)'
        """
        m_re = re.match(r"(.*?), central wavelength (\d+) nm", description)
        if m_re:
            return m_re.group(1) + " (" + m_re.group(2) + " nm)"
        return description

    @staticmethod
    def get_band_short_name(description: str) -> str:
        """
        This method returns only the name of the bands at a chosen resolution.

        Args:
            description: This is the output of the validate_description method.

        Examples:
            >>> desc = validate_description(ds10.descriptions[0])
            >>> desc
            'B4 (665 nm)'
            >>> get_band_short_name(desc)
            'B4'
        """
        if "," in description:
            return description[: description.find(",")]
        if " " in description:
            return description[: description.find(" ")]
        return description[:3]

    def validate(self, data) -> Tuple:
        """
        This method takes the short name of the bands for each separate resolution and
        returns three lists. The validated_bands and validated_indices contain the
        name of the bands and the indices related to them respectively.
        The validated_descriptions is a list of descriptions for each band
        obtained from the validate_description method.

        Args:
            data: The raster file for a specific resolution.

        Examples:
            >>> validated_10m_bands, validated_10m_indices, \
            >>> dic_10m = validate(ds10)
            >>> validated_10m_bands
            ['B4', 'B3', 'B2', 'B8']
            >>> validated_10m_indices
            [0, 1, 2, 3]
            >>> dic_10m
            defaultdict(<class 'str'>, {'B4': 'B4 (665 nm)',
             'B3': 'B3 (560 nm)', 'B2': 'B2 (490 nm)', 'B8': 'B8 (842 nm)'})
        """
        input_select_bands = "B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12"  # type: str
        select_bands = re.split(",", input_select_bands)  # type: List[str]
        validated_bands = []  # type: list
        validated_indices = []  # type: list
        validated_descriptions = defaultdict(str)  # type: defaultdict
        with rasterio.open(data) as d_s:
            for i in range(0, d_s.count):
                desc = self.validate_description(d_s.descriptions[i])
                name = self.get_band_short_name(desc)
                if name in select_bands:
                    select_bands.remove(name)
                    validated_bands += [name]
                    validated_indices += [i]
                    validated_descriptions[name] = desc
        return validated_bands, validated_indices, validated_descriptions

    @staticmethod
    # pylint: disable-msg=too-many-arguments
    def data_final(
        data, term: List, x_mi: int, y_mi: int, x_ma: int, y_ma: int, n_res, scale
    ) -> np.ndarray:
        """
        This method takes the raster file at a specific
        resolution and uses the output of get_max_min
        to specify the area of interest.
        Then it returns an numpy array of values
        for all the pixels inside the area of interest.
        :param data: The raster file for a specific resolution.
        :param term: The validate indices of the
        bands obtained from the validate method.
        :return: The numpy array of pixels' value.
        """
        if term:
            #print(term)
            with rasterio.open(data) as d_s:
                d_final = np.rollaxis(
                    d_s.read(
                        window=Window(
                            col_off=x_mi // scale,
                            row_off=y_mi // scale,
                            width=(x_ma - x_mi + n_res) // scale,
                            height=(y_ma - y_mi + n_res) // scale,
                        )
                    ),
                    0,
                    3,
                )[:, :, term]
        return d_final     

    #Now useless
    """@staticmethod
    def save_output_json(output_jsonfile, output_dir):
        with open(output_dir + "data.json", "w", encoding="utf-8") as f_p:
            f_p.write(json.dumps(output_jsonfile, indent=2))"""

    # pylint: disable-msg=too-many-arguments
    @staticmethod
    def update(data, size_10m: Tuple, model_output: np.ndarray, xmi: int, ymi: int):
        """
        This method creates the proper georeferencing for the output image.

        Args:

            data: The raster file for 10m resolution.
        """
        # Here based on the params.json file, the output image dimension will be calculated.
        out_dims = model_output.shape[2]

        with rasterio.open(data) as d_s:
            p_r = d_s.profile
        new_transform = p_r["transform"] * A.translation(xmi, ymi)
        p_r.update(dtype=rasterio.uint16)
        p_r.update(driver="GTiff")
        p_r.update(width=size_10m[1])
        p_r.update(height=size_10m[0])
        p_r.update(count=out_dims)
        p_r.update(transform=new_transform)
        return p_r

    def assert_input_params(self):
        if not self.clip_to_aoi:
            if self.bounds or self.geometry:
                raise ValueError(
                    "When clip_to_aoi is set to False, bbox, contains and intersects must be set to null.",
                )
        else:
            if (
                self.bounds is None
                and self.geometry is None
            ):
                raise ValueError(
                    "When clip_to_aoi set to True, you MUST define one of bbox, contains or intersect.",
                )
                
    def save_result(
        self,
        model_output,
        output_bands,
        valid_desc,
        output_profile,
        image_name
    ):
        """
        This method saves the feature collection meta data and the
        image with high resolution for desired bands to the provided location.

        Args:
            model_output: The high resolution image.
            output_bands: The associated bands for the output image.
            valid_desc: The valid description of the existing bands.
            output_profile: The georeferencing for the output image.
            output_features: The meta data for the output image.
            image_name: The name of the output image.
        """

        with rasterio.open(image_name, "w", **output_profile) as d_s:
            for b_i, b_n in enumerate(output_bands):
                d_s.write(model_output[:, :, b_i], indexes=b_i + 1)
                d_s.set_band_description(b_i + 1, "SR " + valid_desc[b_n])

    def start(self):
        """
        This method takes the raster data at 10, 20, and 60 m resolutions and by applying
        data_final method creates the input data for the the convolutional neural network.
        It returns 10 m resolution for all the bands in 20 and 60 m resolutions.

        Args:
            input_fc: geojson FeatureCollection of all input images
        """
        self.assert_input_params()
        #output_jsonfile = self.get_final_json()

        #print("Started process...")
        #path_to_input_img = feature["properties"]["up42.data_path"]
        path_to_output_img = Path(self.input_dir).stem + "_superresolution.tif"
        #self.start(path_to_output_img)
        #self.save_output_json(output_jsonfile, self.output_dir)
        #return output_jsonfile
        data_list, image_level = self.get_data()

        for dsdesc in data_list:
            if "10m" in dsdesc:
                if self.clip_to_aoi:
                    xmin, ymin, xmax, ymax, interest_area = self.area_of_interest(
                        dsdesc
                    )
                else:
                    # Get the pixel bounds of the full scene
                    xmin, ymin, xmax, ymax, interest_area = self.get_max_min(
                        0, 0, 20000, 20000, dsdesc
                    )
                #print("Selected pixel region:")
                #print(f"xmin = {xmin}")
                #print(f"ymin = {ymin}")
                #print(f"xmax = {xmax}")
                #print(f"ymax = {ymax}")
                #print(f"The area of selected region = {interest_area}")
            self.check_size(dims=(xmin, ymin, xmax, ymax))

        for dsdesc in data_list:
            if "10m" in dsdesc:
                #print("Selected 10m bands:")
                validated_10m_bands, validated_10m_indices, dic_10m = self.validate(
                    dsdesc
                )
                data10 = self.data_final(
                    dsdesc, validated_10m_indices, xmin, ymin, xmax, ymax, 1, 1
                )
            if "20m" in dsdesc:
                #print("Selected 20m bands:")
                validated_20m_bands, validated_20m_indices, dic_20m = self.validate(
                    dsdesc
                )
                data20 = self.data_final(
                    dsdesc, validated_20m_indices, xmin, ymin, xmax, ymax, 1, 2
                )
            if "60m" in dsdesc:
                #print("Selected 60m bands:")
                validated_60m_bands, validated_60m_indices, dic_60m = self.validate(
                    dsdesc
                )
                data60 = self.data_final(
                    dsdesc, validated_60m_indices, xmin, ymin, xmax, ymax, 1, 6
                )

        validated_descriptions_all = {**dic_10m, **dic_20m, **dic_60m}

        if validated_60m_bands and validated_20m_bands and validated_10m_bands:
            #print("Super-resolving the 60m data into 10m bands")
            sr60_ = dsen2_60(data10, data20, data60, image_level)
            sr60 = sr60_.astype(np.uint16)
            del sr60_, data60
            #print("Super-resolving the 20m data into 10m bands")
            sr20_ = dsen2_20(data10, data20, image_level)
            sr20 = sr20_.astype(np.uint16)
            del sr20_
        else:
            #print("No super-resolution performed, exiting")
            sys.exit(0)

        if self.copy_original_bands:
            sr_final = np.concatenate((data10.astype(np.uint16), sr20, sr60), axis=2)
            validated_sr_final_bands = (
                validated_10m_bands + validated_20m_bands + validated_60m_bands
            )
        else:
            sr_final = np.concatenate((sr20, sr60), axis=2)
            validated_sr_final_bands = validated_20m_bands + validated_60m_bands

        for dsdesc in data_list:
            if "10m" in dsdesc:
                p_r = self.update(dsdesc, data10.shape, sr_final, xmin, ymin)
        filename = os.path.join(self.output_dir, path_to_output_img)

        #print("Now writing the super-resolved bands")
        #print(sr_final)
        #print(validated_sr_final_bands)
        #print(validated_descriptions_all)
        #print(p_r)
        #print(filename)
        self.save_result(
            sr_final,
            validated_sr_final_bands,
            validated_descriptions_all,
            p_r,
            filename
        )
        del sr_final
        gc.collect()
        #print("This is for releasing memory: %s", gc.collect())
        #print("Writing the super-resolved bands is finished.")
        
