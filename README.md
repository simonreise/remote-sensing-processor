
![image](logo_wide.png)

----

![AppVeyor](https://img.shields.io/appveyor/build/simonreise/remote-sensing-processor)
![PyPI](https://img.shields.io/pypi/v/remote-sensing-processor)
![Conda](https://img.shields.io/conda/v/moskovchenkomike/remote-sensing-processor)

RSP is a tool for geospatial raster data processing.

RSP can preprocess Sentinel-2 and Landsat imagery, create raster mosaics, calculate vegetation indices and perform image segmentation tasks.

Read the documentation for more details: https://remote-sensing-processor.readthedocs.io

## Example

Here is an example of some features that RSP provides. Sentinel-2 images are being preprocessed and merged into a mosaic, NDVI of that Sentinel-2 mosaic is calculated. Landcover images are merged into mosaic at the same resolution and projection as Sentinel-2 data. Then Sentinel-2 and landcover data is divided into tiles and UperNet model that predicts landcover based on Sentinel-2 data is trained. This model is used to create landcover map. 
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

# applying min/max normalization to sentinel-2 mosaics (sentinel-2 reflectance values are from 0 to 10 000)
for band in mosaic_sentinel:
	rsp.normalize(band, band, 0, 10000)

# calculating NDVI for sentinel-2 mosaic
ndvi = rsp.calculate_index('NDVI', '/home/rsp_test/mosaics/sentinel/')

# merging landcover files into mosaic 
# and bringing it to resolution and projection of a reference file (one of sentinel mosaic bands)
lcs = glob('/home/rsp_test/landcover/*.tif')
mosaic_landcover = rsp.mosaic(lcs, '/home/rsp_test/mosaics/landcover/', clipper = border, reference_raster = '/home/rsp_test/mosaics/sentinel/B1.tif', nodata = -1)

# cutting sentinel (x) and landcover (y) data to 256x256 px tiles, 
# with random shuffling samples, splitting data into train,
# validation and test subsets in proportion 3 to 1 to 1
# and ignoring tiles with only nodata values
x = mosaic_sentinel
y = mosaic_landcover[0]
x_i, y_i, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata = rsp.segmentation.generate_tiles(x, y, tile_size = 256, shuffle = True, split = [3, 1, 1])
x_train = x_i[0]
x_val = x_i[1]
x_test = x_i[2]
y_train = y_i[0]
y_val = y_i[1]
y_test = y_i[2]
x_train = x_i[0]

# training UperNet that predicts landcover class based on sentinel imagery
model = rsp.segmentation.train(x_train, y_train, x_val, y_val, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, classification = classification, num_classes = num_classes, x_nodata = x_nodata, y_nodata = y_nodata)

# testing model
rsp.segmentation.test(x_test, y_test, model = model)

# mapping landcover based on predictions of our UperNet
y_reference = '/home/rsp_test/mosaics/landcover/landcover01_mosaic.tif'
output_map = '/home/rsp_test/prediction.tif'
rsp.segmentation.generate_map([x_train, x_val, x_test], y_reference, model, output_map, tiles = tiles, samples = samples)
```

## Requirements
To run RSP you need these packages to be installed:
- Python >= 3.8
- Numpy
- GDAL
- Rasterio
- Pyproj
- Shapely
- Fiona
- Geopandas
- Scikit-learn
- Pytorch >= 1.10
- Torchvision >= 0.10
- Lightning
- Transformers

Also you need a Sen2Cor to be installed to process Sentinel-2 data. Required version is 2.11 for Windows and Linux and 2.9 for Mac OS. You should install it via SNAP plugin installer. [Here](http://wiki.awf.forst.uni-goettingen.de/wiki/index.php/Installation_of_SNAP) is the instruction how you can do it.

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


## License
See [LICENSE](LICENSE).

## Thanks to
RSP uses code from some other projects.

Sentinel-2 superresolution is based on [s2-superresolution] by UP42, which is based on [DSen-2] by lanha.

Sentinel-2 atmospheric correction is performed with [Sen2Cor].

Landsat processing module uses code from [Semi-Automatic Classification Plugin](https://fromgistors.blogspot.com/p/semi-automatic-classification-plugin.html).



   [s2-superresolution]: <https://github.com/up42/s2-superresolution>
   [DSen-2]: <https://github.com/lanha/DSen2>
   [Sen2Cor]: <https://step.esa.int/main/snap-supported-plugins/sen2cor/>
   
   

