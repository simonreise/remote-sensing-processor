# Release History

## 0.3 (2025-mm-dd)

### ML module changes
- Now `semantic` and `regression` are separate modules
- Added models from `segmentation_models_pytorch` and `torchgeo` libraries
- Added more sklearn models
- Completely changed the syntax
- Code base is almost completely rewritten
- ML datasets are now stored in `.rspds` format - a custom ML dataset format based on Huggingface Datasets library
- Now any custom Pytorch or Sklearn-based model can be trained
- More metrics available, including custom metrics
- Loss selection available, including custom losses
- Now user can define the augmentations that will be applied to train dataset, including custom augmentations
- Now basic multiscale features can be generated to improve the modeling quality of Sklearn-based models
- Added `band_importance` functions that use SHAP to estimate band importance for the modeling
- Added `confusion_matrix` function that calculate confusion matrix for semantic segmentation models

### Other major changes
- Added `process` function that can clip, reproject, reproject match and change dtype of a single raster
- Added `dem` group of functions: `slope`, `curvature`, `aspect`, `hillshade`
- Added `match_hist` function that matches histograms of two images/datasets
- Added `clip_values` function that clips raster values to a certain range
- Added `denormalize` functions that restore original values from min-max normalized data
- Added `get_normalization_params` functions that retrieve optimal normalization parameters
- Added `zscore` and `dynamicworld` normalization
- Now data is saved with a metadata file in STAC format (can be controlled with `write_stac` argument)
- Almost every function now supports not only file paths, but also STAC Items as inputs
- Most of the functions now can process multi-band datasets and STAC datasets
- Reworked `calculate_index` function, now it supports all the indices supported by `spyndex` library

### Minor changes
- `replace` now supports multiple values replacement via `values` arg
- Added `clip_values` and `nodata` args to `normalize` function
- `input_file` and `output_file` args are renamed to `input_path` and `output_path`
- `process` now accepts `dtype` arg, which will convert input dataset to the requested dtype
- Landsat imagery is now processed by satpy
- Sentinel-2 superresolution models are now stored on Huggingface Hub
- `sen2cor` parameter of `sentinel2` function is now `False`. Sen2Cor support is going to be deprecated in the future.
- The required Sen2Cor version is now 02.12.03
- Multiple performance optimisations

### Deprecations
- `landsat` no longer supports Collection-1 products because they are no longer available to download


## 0.2.2 (2024-04-30)

This update reworks semantic segmentation functions and improves processing speed and stability

- Now uses `xarray`, `dask` and `rioxarray` instead of `numpy` and `rasterio`
- Now stores tiles in zarr containers instead of hdf5
- Syntax, inputs and outputs of all `segmentation` functions are reworked
- Custom `kwargs` can be used when initialising models
- Augmentations can be applied while training with `augment` arg
- Dataset size can be increased by repeating it n times while training with `repeat` arg
- Raster histograms now can be matched while creating mosaic using `match_hist` arg
- Specific value in a raster can be replaced using `replace_value` function
- Nodata value in a raster can be replaced using `replace_nodata` function
- Vector file can be rasterized using `rasterize` function
- Sentinel2 now can be upscaled using resampling algorithm. `superres` arg is renamed to `upscale`, `resample` arg added
- Sentinel2 now can be normalized using `normalize` arg
- Landsat thermal bands now can be normalized using `normalize_t` arg
- `clipper` argument is renamed to `clip`
- `projection` argument is renamed to `crs`


## 0.2.1 (2023-08-23)

- Added `normalize` function that applies min/max normalization to data
- Segmentation `train` and `test` now support multiple datasets input
- Segmentation `train`, `test` and `generate_map` now support multiprocessing
- Added support for more Landsat products
- Various bug fixes

## 0.2 (2023-07-17)

Remote Sensing Processor 0.2 adds image segmentation module

- Added `train` and `test` functions that train and test pytorch and sklearn segmentation models
- `generate_tiles` and `generate map` functions reworked and moved to `rsp.segmentation` module
- Sentinel-2 superresolution algorithm rewritten in pytorch
- `normalized_difference` function renamed to `calculate_index`

## 0.1 (2022-06-13)

This is the first release of Remote Sensing Processor.

It includes Sentinel-2 and Landsat preprocessing, creating raster mosaics, calculating normalized difference indices (for now NDVI only), cutting rasters to tiles and creating maps using pre-trained models.