# Introduction

## Why RSP

The goals of RSP are to:

- Provide a full-cycle workflow for geospatial machine learning - from data processing to training the model and making predictions with it 
- Make geospatial machine learning simple and accessible while keeping it functional and customizable

Usually processing remote sensing data and training geospatial ML models in Python is complicated and need lots of code, because standard GIS libraries like GDAL or Rasterio and machine learning libraries like Scikit-learn or Pytorch provide only low-level functions. To preprocess Landsat of Sentinel image with Rasterio you need to define all the stages of preprocessing: reading data, atmospheric correction, pansharpening, cloud masking, reprojecting and writing result to a file manually with lots of code.

RSP provides high-level functions that automate routine processing operations like remote sensing data preprocessing, merging, calculating vegetation indices, training and testing models. For example, you can preprocess Sentinel-2 image from archive with operations of atmospheric correction, 20- and 60-m bands superresolution, cloud masking and reprojecting to needed projection with one line of code.

Another key idea of RSP is easy pipeline construction, where outputs from one function can be used as inputs to other functions. For example, you can preprocess several Sentinel-2 images with ```sentinel2``` function, then merge preprocessed images with ```mosaic``` function, and then cut merged band rasters into tiles with ```generate_tiles``` function.

```
output_sentinels = rsp.sentinel2(sentinel2_imgs)
x = rsp.mosaic(output_sentinels, '/home/rsp_test/mosaics/sentinel/')
x_tiles, y_tiles = rsp.semantic.generate_tiles(x, y)
```

Also, RSP writes the outputs of the most of the functions to files, which makes possible to resume the interrupted pipeline from the last successful stage, easily return to previous stage or explore the intermediate data in traditional GIS systems. By default, RSP also saves dataset metadata in a STAC format in JSON files alongside the data itself. 

## FAQ

### What exactly does RSP do?

#### General data preprocessing

With `process` you can clip, reproject, match with another raster, fill the gaps in a raster

With `replace_value` you can replace specific value in a raster.

With `replace_nodata` you can replace nodata value in a raster.

With `rasterize` you can rasterize a vector file (shapefile, geopackage, geojson etc.)

With `match_hist` you can match histogram of a raster to a histogram of another raster.

With `clip_values` you can clip the values in a raster to a specific range to remove outliers.

#### Data normalization

`normalize` module is for data normalization.

With `normalize.min_max` you can apply min-max normalization.

With `normalize.z_score` you can apply z-score normalization.

With `normalize.dynamic_world` you can apply log-transform + sigmoid (dynamic world) normalization.

With `denormalize.min_max` you can restore the original values from min-max normalized data.

With `denormalize.z_score` you can restore the original values from z-score normalized data.

With `denormalize.dynamic_world` you can restore the original values from data normalized with dynamic world normalization.

With `get_normalization_params.min_max` you can get min and max values of a raster.

With `get_normalization_params.z_score` you can get mean and standard deviation of a raster.

With `get_normalization_params.percentile` you can get specific percentiles of a raster.

With `get_normalization_params.dynamic_world` you can get parameters for dynamic world normalization.

#### Satellite imagery

With `sentinel2` you can preprocess Sentinel-2 imagery. Preprocessing include upscaling 20- and 60-m bands to 10-m resolution, cloud masking, reprojection, clipping and normalization.

With `landsat` you can preprocess Landsat imagery. Preprocessing include DOS-1 atmospheric correction, cloud masking, pansharpening for Landsat 7 and 8, calculating temperature from thermal band, reprojection and clipping.

#### Vegetation indices

With `calculate_index` you can calculate normalized difference indexes like NDVI.

#### DEM

With `dem.aspect` you can calculate aspect from a DEM.

With `dem.slope` you can calculate slope from a DEM.

With `dem.curvature` you can calculate curvature from a DEM.

With `dem.hillshade` you can calculate hillshade from a DEM.

#### Mosaics

With `mosaic` you can merge several rasters (or multi-band products) into mosaic, fill the gaps in it, match their histograms, match it to a reference raster and clip it to ROI.

#### Semantic segmentation

With `semantic` module you can train semantic segmentation model and use it for predictions.

With `semantic.generate_tiles` you can generate an ML-ready dataset ,which includes cutting rasters into tiles, splitting data to subdatasets and shuffling the samples.

With `semantic.train` you can train a machine learning model for semantic segmentation using generated tiles.

With `semantic.test` you can test a semantic segmentation model.

With `semantic.generate_map` you can create map from predictions of pre-trained segmentation model.

With `semantic.band_importance` you can estimate importance of different bands for the modeling.

#### Regression

With `regression` module you can train regression model and use it for predictions.

With `regression.generate_tiles` you can generate an ML-ready dataset ,which includes cutting rasters into tiles, splitting data to subdatasets and shuffling the samples.

With `regression.train` you can train a machine learning model for regression using generated tiles.

With `regression.test` you can test a regression model.

With `regression.generate_map` you can create map from predictions of pre-trained regression model.

With `semantic.band_importance` you can estimate importance of different bands for the modeling.

### Are you planning to add preprocessing of other imagery types (Sentinel-1, MODIS, GEOS etc.)?

Yes, but it is a long-term goal. First, I will focus on improving current functionality and adding other ML tasks (object detection, panoptic segmentation etc.) Also, you can contribute by adding your code!

### I keep running into memory errors.

RSP is mostly optimized for performance rather than for memory efficiency. Despite using Dask arrays at the backend, it is still likely to fail when processing data that does not fit into memory. So, I highly recommend using a swap file on Linux and Mac or a pagefile on Windows. Also, it is planned to add Dask cluster support, but, if you read this, I still have no success with it.

### I got a pytorch-related error.

You need to have Nvidia GPU in your PC to run pytorch models. Actually, it should also work on CPU, but the processing can be very slow.

If you have GPU and received this error, try to re-install `pytorch` using [official guide](https://pytorch.org/get-started/locally/). Or try another pytorch version.

If you are get such error while Sentinel-2 processing, try to run `sentinel2` with `upscale = 'resample'` or `upscale = None`.

### I want to report an error / suggest adding a new feature

Feel free to open new tickets at https://github.com/simonreise/remote-sensing-processor/issues anytime.

### How can I cite RSP?

If you use RSP in a scientific publication, we would appreciate citations: https://doi.org/10.5281/zenodo.11091321

### I got error 'Sen2Cor not working. Is it installed correctly?'.

Looks like you did not install Sen2Cor. RSP uses Sen2Cor which is installed via SNAP plugin installer. [Here](http://wiki.awf.forst.uni-goettingen.de/wiki/index.php/Installation_of_SNAP) is the instruction how you can do it. If you don't want to install SNAP, you can manually install [Sen2Cor 2.11](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-11/) or [Sen2Cor 2.9](https://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-9/) to `%HOME%/.snap/auxdata/`. If you installed Sen2Cor correctly, but it still does not work, you can set flag `sen2cor = False`.

Also, as Level-2 imagery is now widely available, we will soon discontinue support of Sen2Cor. 

## License

RSP is an open source software distributed under [GNU General Public License v3.0](https://github.com/simonreise/remote-sensing-processor/blob/master/LICENSE)