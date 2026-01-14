# Remote Sensing Processor

RSP is a tool for geospatial raster data processing and machine learning.

RSP can preprocess satellite imagery and DEM data, create raster mosaics, calculate vegetation indices and perform image segmentation and regression tasks.

The goals of RSP are to:

- Provide a full-cycle workflow for geospatial machine learning - from data processing to training the model and making predictions with it 
- Make geospatial machine learning simple and accessible while keeping it functional and customizable

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
- Calculate a confusion matrix

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
mosaic_sentinel = rsp.mosaic(
	output_landsats, 
	"/home/rsp_test/mosaics/landsat/",
    fill_nodata=True,
	clip=border,
	crs="EPSG:4326",
	nodata_order=True,
)

# Calculating NDVI for Landsat mosaic
ndvi = rsp.calculate_index("NDVI", "/home/rsp_test/mosaics/landsat/")

# Merging DEM files into mosaic 
# and matching it to resolution and projection of a reference file (one of Sentinel mosaic bands)
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

# Rasterizing landcover type classification shapefile
# and matching it to resolution and projection of a reference file (one of Sentinel mosaic bands)
landcover_shp = "/home/rsp_test/landcover/types.shp"

# Preparing data for semantic segmentation
# Cut Sentinel and DEM (training data) and landcover (target variable) data to 256x256 px tiles, 
# random shuffle samples, split data into train, validation and test subsets in proportion 3 to 1 to 1
# If target variable is a vector, it can be automatically rasterized
x = mosaic_sentinel + dem + slope
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

## Citation
If you use RSP in a scientific publication, we would appreciate citations: https://doi.org/10.5281/zenodo.11091321


```{eval-rst}
.. toctree::
   :maxdepth: 2

   intro
   install
   quickstart
   api
```