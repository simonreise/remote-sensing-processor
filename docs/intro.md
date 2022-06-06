# Introduction

## Why RSP

Usually preprocessing remote sensing data in Python is complicated and need lots of code, because standard GIS libraries like GDAL and Rasterio provide only low-level functions like reading, writing, reprojecting and clipping rasters. To preprocess Landsat of Sentinel image with Rasterio you need to define all the stages of preprocessing: reading data, atmospheric correction, pansharpening, cloud masking, reprojecting and writing result to a file manually with lots of coding.

RSP provides high-level functions that automate routine processing operations like remote sensing data preprocessing, merging and calculating vegetation indices. For example, you can preprocess Sentinel-2 image from L1 zip archive with operations of atmospheric correction, 20- and 60-m bands superresolution, cloud masking and reprojecting to needed projection with one line of code.

Another key idea of RSP is easy pipeline buildings, where outputs from one function can be used as inputs to other functions. For example, you can preprocess several Sentinel-2 images with ```sentinel2``` function, then megre preprocessed images with ```mosaic``` function, and then cut merged band rasters into tiles with ```generate_tiles``` function.
```
output_sentinels = rsp.sentinel2(sentinel2_imgs)
x = rsp.mosaic(output_sentinels, '/home/rsp_test/mosaics/sentinel/')
x_i, y_i, tiles, samples = rsp.generate_tiles(x, y)
```

## FAQ

### What exactly does RSP do?

With `sentinel2` you can preprocess Sentinel-2 imagery. Preprocessing include upgrading L1 product to L2 (mostly atmospheric correction), upscaling 20- and 60-m bands to 10-m resolution, cloud masking, reprojection and clipping.

With `landsat` you can preprocess Landsat imagery. Preprocessing include DOS-1 atmospheric correction, cloud masking, pansharpening for Landsat 7 and 8, calculating temperature from thermal band, reprojection and clipping.

With `mosaic` you can merge several rasters (or Sentinel-2 or Landsat products) into mosaic, fill the gaps in it and clip it to ROI.

With `normalized_difference` you can calculate normalized difference indexes like NDVI.

With `generate_tiles` you can cut rasters into tiles that can be used e.g. for convolutional neural network (CNN) training and with `generate_map` you can create map from predictions of pre-trained CNN.

### Are you planning to add preprocessing of other imagery types (Sentinel-1, MODIS, GEOS etc.)?

Well yes but actually no. This library is a compilation of jupyter notebooks I wrote for imagery types i needed to preprocess. If sometime I'll need to preprocess any other imagery type - I'll add function for it in RSP. Also, you can contrubute by adding your code!

### I recieve error 'Sen2Cor not working. Is it installed correctly?'.

Looks like you did not install Sen2Cor. RSP uses Sen2Cor which is installed via SNAP plugin installer. [Here](http://wiki.awf.forst.uni-goettingen.de/wiki/index.php/Installation_of_SNAP) is the instruction how you can do it. If you don't want to install SNAP, you can manually install [Sen2Cor 2.8](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor_v2-8/) to `%HOME%/.snap/auxdata/`. If you installed Sen2Cor correctly and it still does not work, you can set flag `sen2cor = False`.

### I tried to preprocess Sentinel-2 and recieved error 'The Conv2D op currently only supports the NHWC tensor format on the CPU'.

Looks like you does not have tensorflow-supported Nvidia GPU in your PC. It is needed for 20- and 60-m bands upscaling. Try to run `sentinel2` with `superres = False` flag.

If you have tensorflow-supported GPU and recieved this error, try to uninstall `tensorflow` and install `tensorflow-gpu` instead. Or try another tensorflow version. It was definetely working with tensorflow-gpu 2.6.0, cudatoolkit 11.3.1 and cudnn 8.2.1.32 installed via conda. 

### I want to report an error / suggest adding a new feature

Feel free to open new tickets at https://github.com/simonreise/remote-sensing-processor/issues anytime.

## License

RSP is an open source software distributed under [GNU General Public License v3.0](https://github.com/simonreise/remote-sensing-processor/blob/master/LICENSE)