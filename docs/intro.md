# Introduction

## Why RSP

Usually preprocessing remote sensing data in Python is complicated and need lots of code, because standard GIS libraries like GDAL and Rasterio provide only low-level functions like reading, writing, reprojecting and clipping rasters. To preprocess Landsat of Sentinel image with Rasterio you need to define all the stages of preprocessing: reading data, atmospheric correction, pansharpening, cloud masking, reprojecting and writing result to a file manually with lots of coding.

RSP provides high-level functions that automate routine processing operations like remote sensing data preprocessing, merging and calculating vegetation indices. For example, you can preprocess Sentinel-2 image from L1 zip archive with operations of atmospheric correction, 20- and 60-m bands superresolution, cloud masking and reprojecting to needed projection with one line of code.

Another key idea of RSP is easy pipeline construction, where outputs from one function can be used as inputs to other functions. For example, you can preprocess several Sentinel-2 images with ```sentinel2``` function, then megre preprocessed images with ```mosaic``` function, and then cut merged band rasters into tiles with ```generate_tiles``` function.
```
output_sentinels = rsp.sentinel2(sentinel2_imgs)
x = rsp.mosaic(output_sentinels, '/home/rsp_test/mosaics/sentinel/')
x_i, y_i, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata = rsp.segmentation.generate_tiles(x, y)
```

## FAQ

### What exactly does RSP do?

With `sentinel2` you can preprocess Sentinel-2 imagery. Preprocessing include upgrading L1 product to L2 (mostly atmospheric correction), upscaling 20- and 60-m bands to 10-m resolution, cloud masking, reprojection and clipping.

With `landsat` you can preprocess Landsat imagery. Preprocessing include DOS-1 atmospheric correction, cloud masking, pansharpening for Landsat 7 and 8, calculating temperature from thermal band, reprojection and clipping.

With `mosaic` you can merge several rasters (or Sentinel-2 or Landsat products) into mosaic, fill the gaps in it and clip it to ROI.

With `calculate_index` you can calculate normalized difference indexes like NDVI.

With `segmentation` module you can train segmentation model and use it for predictions.

With `segmentation.generate_tiles` you can cut rasters into tiles that can be used e.g. for convolutional neural network (CNN) training and

With `segmentation.train` you can train a machine learning model for image segmentation using generated tiles.

With `segmentation.test` you can test segmentation model.

With `segmentation.generate_map` you can create map from predictions of pre-trained segmentation model.

### Are you planning to add preprocessing of other imagery types (Sentinel-1, MODIS, GEOS etc.)?

Well yes but actually no. This library is a compilation of jupyter notebooks I wrote for imagery types I needed to preprocess. If sometime I'll need to preprocess any other imagery type - I'll add function for it in RSP. Also, you can contrubute by adding your code!

### I got error 'Sen2Cor not working. Is it installed correctly?'.

Looks like you did not install Sen2Cor. RSP uses Sen2Cor which is installed via SNAP plugin installer. [Here](http://wiki.awf.forst.uni-goettingen.de/wiki/index.php/Installation_of_SNAP) is the instruction how you can do it. If you don't want to install SNAP, you can manually install [Sen2Cor 2.11](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-11/) or [Sen2Cor 2.9](https://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-9/) to `%HOME%/.snap/auxdata/`. If you installed Sen2Cor correctly and it still does not work, you can set flag `sen2cor = False`.

### I got a pytorch-related error.

You need to have Nvidia GPU in your PC to run pytorch models. Actually, it must run even on CPU, but the processing can be very slow.

If you have GPU and recieved this error, try to re-install `pytorch` using [official guide](https://pytorch.org/get-started/locally/). Or try another pytorch version.

If you are get such error while Sentinel-2 processing, try to run `sentinel2` with `superres = False` flag.

### I want to report an error / suggest adding a new feature

Feel free to open new tickets at https://github.com/simonreise/remote-sensing-processor/issues anytime.

## License

RSP is an open source software distributed under [GNU General Public License v3.0](https://github.com/simonreise/remote-sensing-processor/blob/master/LICENSE)