# Remote Sensing Processor
----
RSP is a tool for geospatial raster data processing.

Most of remote sensing data like Sentinel of Landsat imagery needs to be preprocessed before using. RSP can preprocess Sentinel-2 and Landsat imagery, create raster mosaics, calculate vegetation indices, cut rasters into tiles.

Here is an example of some features that RSP provides. Sentinel-2 images are being preprocessed and merged into a mosaic, NDVI of that Sentinel-2 mosaic is calculated. Landcover images are merged into mosaic at the same resolution and projection as Sentinel-2 data. Then Sentinel-2 and landcover data is divided into tiles and U-Net model that predicts landcover based on Sentinel-2 data is trained. This model is used to create landcover map. 
```
from glob import glob
import remote_sensing_processor as rsp


# getting a list of sentinel-2 images
sentinel2_imgs = glob('/home/rsp_test/sentinels/*.zip')

# preprocessing sentinel-2 images
# it includes converting L1 products to L2, superresolution for 20 and 60m bands and cloud masking
output_sentinels = rsp.sentinel2(sentinel2_imgs)

# merging sentinel-2 images into one mosaic 
# in order from images with most nodata values on bottom to images with less nodata on top,
# clipping it to the area of interest and reprojecting to the proj we need
border = '/home/rsp_test/border.gpkg'
mosaic_sentinel = rsp.mosaic(output_sentinels, '/home/rsp_test/mosaics/sentinel/', clipper = border, projection = 'EPSG:4326', nodata_order = True)

# calculating NDVI for sentinel-2 mosaic
ndvi = rsp.normalized_difference('NDVI', '/home/rsp_test/mosaics/sentinel/')

# merging landcover files into mosaic 
#and bringing it to resolution and projection of a reference file (one of sentinel mosaic bands)
lcs = glob('/home/rsp_test/landcover/*.tif')
mosaic_landcover = rsp.mosaic(lcs, '/home/rsp_test/mosaics/landcover/', clipper = border, reference_raster = '/home/rsp_test/mosaics/sentinel/B1.tif', nodata = -1)

# cutting sentinel (x) and landcover (y) data to 256x256 px tiles, 
# with random shuffling samples, splitting data into train,
# validation and test subsets in proportion 3 to 1 to 1
# and ignoring tiles with only nodata values
x = mosaic_sentinel
y = mosaic_landcover[0]
x_i, y_i, tiles, samples = rsp.generate_tiles(x, y, num_classes = 11, tile_size = 256, shuffle = True, split = [3, 1, 1], nodata = -1)
x_train = x_i[0]
x_val = x_i[1]
x_test = x_i[2]
y_train = y_i[0]
y_val = y_i[1]
y_test = y_i[2]

# training U-Net that predicts landcover class based on sentinel imagery
from tensorflow import keras
# =======================
# here must be u-net code
# =======================
model.fit(x_train, y_train, batch_size = 16, epochs = 20, validation_data = (x_val, y_val), callbacks = callbacks)
# testing_model
model.evaluate(x_test, y_test)

# mapping landcover based on predictions of our U-Net
y_reference = '/home/rsp_test/mosaics/landcover/landcover01_mosaic.tif'
output_map = '/home/rsp_test/prediction.tif'
rsp.generate_map([x_train, x_val, x_test], y_reference, model, output_map, tiles = tiles, samples = samples)
```

.. toctree::
   :maxdepth: 2

   intro
   install
   quickstart
   api