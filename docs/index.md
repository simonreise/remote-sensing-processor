# Remote Sensing Processor

RSP is a tool for geospatial raster data processing.

RSP can preprocess Sentinel-2 and Landsat imagery, create raster mosaics, calculate vegetation indices and perform image segmentation tasks.

Here is an example of some features that RSP provides. We will perform a simple task - train a semantic segmentation model that will predict landcover class using Sentinel-2 and DEM data.

First, we need to preprocess data. We preprocess Sentinel-2 images and merge them into a mosaic, then we calculate NDVI of that Sentinel-2 mosaic. We also merge DEM images into mosaic, match it to the same resolution and projection as Sentinel-2 data and normalize its values. We rasterize landcover shape file and match it to the same resolution and projection as Sentinel-2 data. 

Then we prepare data to semantic segmentation model training. We use Sentinel-2 and DEM data as training data and landcover data as a target variable. We cut our data into small tiles and split it into train, validation and test subsets. Then we train and test UperNet model that predicts landcover based on Sentinel-2 and DEM data. Finally, we use this model to create a landcover map. 
```python
from glob import glob
import remote_sensing_processor as rsp


# Getting a list of Sentinel-2 images
sentinel2_imgs = glob('/home/rsp_test/sentinels/*.zip')

# Preprocessing sentinel-2 images
# It includes converting L1 products to L2, superresolution for 20 and 60m bands and cloud masking
# We also normalize data values to range from 0 to 1
output_sentinels = rsp.sentinel2(sentinel2_imgs, normalize = True)

# Merging sentinel-2 images into one mosaic 
# in order from images with less nodata pixels on top to images with most nodata on bottom
# clipping it to the region of interest and reprojecting to the crs we need
border = '/home/rsp_test/border.gpkg'
mosaic_sentinel = rsp.mosaic(output_sentinels, '/home/rsp_test/mosaics/sentinel/', clip = border, projection = 'EPSG:4326', nodata_order = True)

# Calculating NDVI for Sentinel-2 mosaic
ndvi = rsp.calculate_index('NDVI', '/home/rsp_test/mosaics/sentinel/')

# Merging DEM files into mosaic 
# and matching it to resolution and projection of a reference file (one of Sentinel mosaic bands)
dems = glob('/home/rsp_test/dem/*.tif')
mosaic_dem = rsp.mosaic(dems, '/home/rsp_test/mosaics/dem/', clip = border, reference_raster = '/home/rsp_test/mosaics/sentinel/B1.tif', nodata = 0)

# Applying min/max normalization to DEM mosaic (heights in our region of interest are in range from 100 to 1000)
rsp.normalize(mosaic_dem[0], mosaic_dem[0], 100, 1000)

# Rasterizing landcover type classification shapefile
# and matching it to resolution and projection of a reference file (one of Sentinel mosaic bands)
landcover_shp = '/home/rsp_test/landcover/types.shp'
landcover = '/home/rsp_test/landcover/types.tif'
rsp.rasterize(landcover_shp, reference_raster = '/home/rsp_test/mosaics/sentinel/B1.tif', value = "type", output_file = landcover)

# Preparing data for semantic segmentation
# Cut Sentinel and DEM (training data) and landcover (target variable) data to 256x256 px tiles, 
# random shuffle samples, split data into train, validation and test subsets in proportion 3 to 1 to 1
x = mosaic_sentinel + mosaic_dem
y = landcover
x_tiles, y_tiles = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1], split_names = ['train', 'val', 'test'])

# Training UperNet that predicts landcover class based on Sentinel and DEM
train_ds = [x_tiles, y_tiles[0], 'train']
val_ds = [x_tiles, y_tiles[0], 'val']
model = rsp.segmentation.train(train_ds, val_ds, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 100)

# Testing model
test_ds = [x_tiles, y_tiles[0], 'test']
rsp.segmentation.test(test_ds, model = model)

# Mapping landcover using predictions of our UperNet
reference = landcover
output_map = '/home/rsp_test/prediction.tif'
rsp.segmentation.generate_map(x_tiles, y_tiles[0], reference, model, output_map)
```

```{eval-rst}
.. toctree::
   :maxdepth: 2

   intro
   install
   quickstart
   api
```