
![image](docs/_static/logo/logo_wide.png)

----

[![Build status](https://ci.appveyor.com/api/projects/status/usca6y014oakdtj2?svg=true)](https://ci.appveyor.com/project/simonreise/remote-sensing-processor)
[![PyPI](https://img.shields.io/pypi/v/remote-sensing-processor)](https://pypi.org/project/remote-sensing-processor/)
[![Conda](https://img.shields.io/conda/v/moskovchenkomike/remote-sensing-processor)](https://anaconda.org/moskovchenkomike/remote-sensing-processor)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11091321.svg)](https://doi.org/10.5281/zenodo.11091321)

RSP is a tool for geospatial raster data processing and machine learning.

RSP can preprocess satellite imagery and DEM data, create raster mosaics, calculate vegetation indices and perform image segmentation and regression tasks.

The goals of RSP are to:

- Provide a full-cycle workflow for geospatial machine learning - from data processing to training the model and making predictions with it 
- Make geospatial machine learning simple and accessible while keeping it functional and customizable

Read the documentation for more details: https://remote-sensing-processor.readthedocs.io

## Key features

### General data preprocessing

- Clip, reproject, match with another raster, fill the gaps
- Normalize raster data
- Match histograms of two rasters
- Rasterize vector data
- Replace specific values and nodata value
- Clip data values

### Satellite imagery
- Sentinel-2: mask clouds, upscale 20- and 60-m bands, clip, reproject, normalize
- Landsat: mask clouds, convert to reflectance, pansharpen, clip, reproject, normalize

### Vegetation indices

- Calculate all vegetation indices that are supported in [Awesome Spectral Indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices?tab=readme-ov-file#spectral-indices-by-application-domain)

### DEM

Calculate:
- Aspect
- Slope
- Curvature
- Hillshade

## Mosaic

- Generating mosaics from several single-band or multi-band rasters 

## Machine learning

- Generate ML-ready datasets from your custom geospatial data - cut data into tiles, split to train / validation / test subdatasets, random shuffle the data
- Train both Deep Learning and classical ML models, including custom models
- Apply data augmentation techniques, including custom augmentations
- Estimate model performance with different metrics (custom metrics are supported too) and save the logs
- Use trained ML models to make predictions and generate maps out of them
- Estimate band importance for the modeling

### Semantic segmentation

Semantic segmentation module supports models from:
- HuggingFace Transformers
- Segmentation-Models-Pytorch
- Torchgeo
- Custom Torch models
- Scikit-Learn
- XGBoost
- Custom SKlearn-like models

### Regression

Semantic segmentation module supports models from:
- HuggingFace Transformers
- Segmentation-Models-Pytorch
- Torchgeo
- Custom Torch models
- Scikit-Learn
- XGBoost
- Custom SKlearn-like models

## Example

Here is an example of some features that RSP provides. We will perform a simple task - train a semantic segmentation model that will predict landcover class using Landsat and DEM data.

First, we need to preprocess data. We preprocess Landsat images and merge them into a mosaic, then we calculate NDVI of that Landsat mosaic. We also merge DEM images into mosaic, match it to the same resolution and projection as Landsat data, normalize its values and calculate slope. We rasterize landcover shape file and match it to the same resolution and projection as Landsat data. 

Then we prepare data to semantic segmentation model training. We use Landsat and DEM data as training data and landcover data as a target variable. We cut our data into small tiles and split it into train, validation and test subsets. Then we train and test UperNet model that predicts landcover based on Landsat and DEM data. Finally, we use this model to create a landcover map. 

```python
from glob import glob
import remote_sensing_processor as rsp


# Getting a list of Landsat images
landsat_imgs = glob("/home/rsp_test/landsats/*.zip")

# Preprocessing Landsat images
# It includes converting to reflectance, pansharpening and cloud masking
# We also normalize data values to range from 0 to 1
output_landsats = rsp.landsat(landsat_imgs, normalize=True)

# Merging Landsat images into one mosaic 
# in order from images with less nodata pixels on top to images with most nodata on bottom
# filling the gaps, clipping it to the region of interest and reprojecting to the crs we need
border = "/home/rsp_test/border.gpkg"
mosaic_landsat = rsp.mosaic(
	output_landsats, 
	"/home/rsp_test/mosaics/landsat/",
    fill_nodata=True,
	clip=border,
	crs="EPSG:4326",
	nodata_order=True,
)

# Calculating NDVI for Landsat mosaic
ndvi = rsp.calculate_index("NDVI", mosaic_landsat)

# Merging DEM files into mosaic 
# and matching it to resolution and projection of a reference file (one of Landsat mosaic bands)
dems = glob("/home/rsp_test/dem/*.tif")
mosaic_dem = rsp.mosaic(
	dems, 
	"/home/rsp_test/mosaics/dem/", 
	clip=border, 
	reference_raster="/home/rsp_test/mosaics/landsat/B1.tif", 
	nodata=0,
)

# Applying min/max normalization to DEM mosaic (heights in our region of interest are in range from 100 to 1000)
dem = rsp.normalize.min_max(
    mosaic_dem[0],
    minimum=100,
    maximum=1000,
)

# Calculate slope from DEM
slope = "/home/rsp_test/dem/slope.tif"
slope = rsp.dem.slope(
    mosaic_dem,
    output_path=slope,
    normalize=True,
)

# Preparing data for semantic segmentation
# Cut Landsat and DEM (training data) and landcover (target variable) data to 256x256 px tiles, 
# random shuffle samples, split data into train, validation and test subsets in proportion 3 to 1 to 1
# If target variable is a vector, it can be automatically rasterized
x = mosaic_landsat + dem + slope
landcover_shp = "/home/rsp_test/landcover/types.shp"
y = {"name": "landcover", "path": landcover_shp, "value": "type"}
dataset = rsp.semantic.generate_tiles(
	x, 
	y,
    "/home/rsp_test/model/landcover_dataset.rspds",
	tile_size=256,
	shuffle=True,
	split={"train": 3, "validation": 1, "test": 1},
)

# Training UperNet that predicts landcover class based on Landsat and DEM
train_ds = {"path": dataset, "sub": "train", "y": "landcover"}
val_ds = {"path": dataset, "sub": "validation", "y": "landcover"}
model = rsp.semantic.train(
	train_ds,
	val_ds,
    model_file="/home/rsp_test/model/upernet.ckpt",
	model="UperNet", 
	backbone="ConvNeXTV2",
	epochs={"max_epochs":100, "early_stopping": True, "patience": 10},
    augment=True,
    num_workers="auto",
)

# Testing model against several metrics
test_ds = {"path": dataset, "sub": "test", "y": "landcover"}
rsp.semantic.test(
    test_ds,
    model=model,
    metrics=[
        {"name": "accuracy_macro", "log": "step"},
        {"name": "accuracy_micro", "log": "verbose"},
        {"name": "f1_macro", "log": "step"},
        {"name": "f1_micro", "log": "verbose"},
        {"name": "precision_macro", "log": "step"},
        {"name": "precision_micro", "log": "step"},
        {"name": "recall_macro", "log": "step"},
        {"name": "recall_micro", "log": "step"},
        {"name": "generalized_dice_score", "log": "epoch"},
        {"name": "mean_iou", "log": "verbose"},
    ],
    num_workers="auto",
)

# Mapping landcover using predictions of our UperNet
whole_ds = {"path": dataset, "sub": "all", "y": "landcover"}
output_map = "/home/rsp_test/prediction.tif"
rsp.semantic.generate_map(whole_ds, model, output_map)
```

## Installation

From PyPI:
```
pip install remote-sensing-processor
```
From Conda:
```
conda install -c moskovchenkomike remote-sensing-processor
```
From source:
```
git clone https://github.com/simonreise/remote-sensing-processor
cd remote-sensing-processor
pip install .
```

If you have a GPU that supports CUDA we strongly recommend you to install Pytorch version that is built with CUDA support before installing RSP. You can find out how to do it on [Pytorch official site](https://pytorch.org/get-started/locally/).

> :warning: This package is still in early development stage, so its API can change significantly, sometimes without backward compatibility. Consider this before updating the package.

## License
See [LICENSE](LICENSE).

## Citation
If you use RSP in a scientific publication, we would appreciate citations: https://doi.org/10.5281/zenodo.11091321

## Credits

RSP relies on many awesome lower-level Python libraries.

- [NumPy](https://github.com/numpy/numpy), [Xarray](https://github.com/pydata/xarray) and [Dask](https://github.com/dask/dask) - for array processing and parallel computations
- [Rasterio](https://github.com/rasterio/rasterio), [Rioxarray](https://github.com/corteva/rioxarray) and [ODC-Geo](https://github.com/opendatacube/odc-geo) - for geospatial data processing
- [SatPy](https://github.com/pytroll/satpy) - for satellite imagery pre-processing
- [Xarray-Spatial](https://github.com/makepath/xarray-spatial) - for DEM processing
- [Spyndex](https://github.com/awesome-spectral-indices/spyndex) - for vegetation indices calculation
- [Geocube](https://github.com/corteva/geocube) - to rasterize vectors
- [GeoPandas](https://github.com/geopandas/geopandas) - for vector data loading
- [PySTAC](https://github.com/stac-utils/pystac) and [STACTools](https://github.com/stac-utils/stactools) - for STAC handling
- [HuggingFace Datasets](https://github.com/huggingface/datasets) - for ML datasets generation, processing and loading
- [XBatcher](https://github.com/xarray-contrib/xbatcher) - to cut data into tiles
- [Scikit-Image](https://github.com/scikit-image/scikit-image) - to generate multiscale basic features
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) and [XGBoost](https://github.com/dmlc/xgboost) - for classical ML models
- [PyTorch](https://github.com/pytorch/pytorch) and [Lightning](https://github.com/Lightning-AI/lightning) - for Deep Learinig models training
- [Torchvision](https://github.com/pytorch/vision) - for data augmentation and DL vision models
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - for vision transformers
- [Segmentation-Models-Pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) - for DL vision models and custom losses
- [TorchGeo](https://github.com/torchgeo/torchgeo) - for geospatial DL models
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) - metrics for estimating modeling quality
- [SHAP](https://github.com/shap/shap) - for band importance estimation

RSP uses code from some other projects.

Sentinel-2 superresolution is based on [s2-superresolution](https://github.com/up42/s2-superresolution) by UP42, which is based on [DSen-2](https://github.com/lanha/DSen2) by lanha.
