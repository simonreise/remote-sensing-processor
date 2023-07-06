# Quickstart

Remote sensing data can be used for many purposes - for vegetation, climate, soil and human impact analysis. RSP can prepare raw remote sensing data for analysis.
 
Here is an example of some features that RSP provides. In this example we will use Sentinel-2 and ESA landcover data to create a model for land cover classification. 

Sentinel-2 images are being preprocessed and merged into a mosaic, NDVI of that Sentinel-2 mosaic is calculated. Landcover images are merged into mosaic at the same resolution and projection as Sentinel-2 data. Then Sentinel-2 and landcover data is divided into tiles and U-Net model that predicts landcover based on Sentinel-2 data is trained. This model is used to create landcover map. 

## Importing RSP

Here we import Remote Sensing Processor
```
import remote_sensing_processor as rsp
```

## Sentinel-2 preprocessing

We have 6 Sentinel-2 images that cover our region of interest.
```
>>> from glob import glob
>>> sentinel2_imgs = glob('/home/rsp_test/sentinels/*.zip')
>>> print(sentinel2_imgs)
['/home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626.zip',
 '/home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626.zip',
 '/home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626.zip',
 '/home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027.zip',
 '/home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624.zip',
 '/home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041.zip']
```
We need to preprocess these images. Preprocessing includes converting L1 products to L2, superresolution for 20 and 60m bands and cloud masking.

First step of preprocessing is converting L1 raw product to atmospherically corrected L2 product. It is done with Sen2Cor algorythm.
![Atmospheric correction](https://www.researchgate.net/publication/339708639/figure/fig5/AS:865545297620992@1583373488624/Example-of-Sentinel-2A-level1-Top-Of-Atmosphere-TOA-reflectance-RGB-composition-and_W640.jpg)

Image from [here](https://doi.org/10.3390/rs12050833)

Second step is upscaling 20-m and 60-m bands to 10-m resolution with superresolution model.

![Superresolution](https://ars.els-cdn.com/content/image/1-s2.0-S0924271618302636-gr1.jpg)

Image from [here](https://doi.org/10.1016/j.isprsjprs.2018.09.018)

Final step is cloud masking. It is done with cloud mask file from sentinel image, surface type map generated during Sen2Cor L2 product generation and with band 1 and 9 filters.

Sentinel image before masking

![Before](sentinel-raw.png)

Sentinel image after cloud masking.

![Before](sentinel-masked.png)

`sentinel2` function can take list of images as input and process all of them one by one.

```
>>> output_sentinels = rsp.sentinel2(sentinel2_imgs)
Preprocessing of /home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626.zip completed
Preprocessing of /home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626.zip completed
Preprocessing of /home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626.zip completed
Preprocessing of /home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027.zip completed
Preprocessing of /home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624.zip completed
Preprocessing of /home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041.zip completed
>>> print(output_sentinels)
['/home/rsp_test/sentinels/L1C_T42VWR_A032192_20210821T064626/',
 '/home/rsp_test/sentinels/L1C_T42WXS_A032192_20210821T064626/',
 '/home/rsp_test/sentinels/L1C_T43VCL_A032192_20210821T064626/',
 '/home/rsp_test/sentinels/L1C_T43VDK_A031391_20210626T063027/',
 '/home/rsp_test/sentinels/L1C_T43VDL_A023312_20210823T063624/',
 '/home/rsp_test/sentinels/L1C_T43VDL_A031577_20210709T064041/']
```

Function returns list of folders with preprocessed images.

## Merging Sentinel-2 images
In this stage preprocessed Sentinel-2 images are being merged into one mosaic. This function can merge not only single-band images, but also multi-band imagery like Sentinel-2.  `clipper` argument is a path to a file with a border of our region of interest which is used to clip mask, `projection` is a CRS we need, and `nodata_order` is to merge images in order from images with most nodata values on bottom (they usually are most distorted and cloudy) to images with less nodata on top (they are usually clear).
```
>>> border = '/home/rsp_test/border.gpkg'
>>> mosaic_sentinel = rsp.mosaic(output_sentinels, '/home/rsp_test/mosaics/sentinel/', clipper = border, projection = 'EPSG:4326', nodata_order = True)
Processing completed
>>> print(mosaic_sentinel)
['/home/rsp_test/mosaics/sentinel/B1.tif',
 '/home/rsp_test/mosaics/sentinel/B2.tif',
 '/home/rsp_test/mosaics/sentinel/B3.tif',
 '/home/rsp_test/mosaics/sentinel/B4.tif',
 '/home/rsp_test/mosaics/sentinel/B5.tif',
 '/home/rsp_test/mosaics/sentinel/B6.tif',
 '/home/rsp_test/mosaics/sentinel/B7.tif',
 '/home/rsp_test/mosaics/sentinel/B8.tif',
 '/home/rsp_test/mosaics/sentinel/B8A.tif',
 '/home/rsp_test/mosaics/sentinel/B9.tif',
 '/home/rsp_test/mosaics/sentinel/B11.tif',
 '/home/rsp_test/mosaics/sentinel/B12.tif']
```
The function returns list of band mosaics.


## Calculating NDVI for sentinel-2 mosaic

Normalized difference function can automatically select bands for calculating NDVI based on Sentinel-2 image, we can just give it index name and a folder where bands are stored.
```
>>> ndvi = rsp.calculate_index('NDVI', '/home/rsp_test/mosaics/sentinel/')
>>> print(ndvi)
'/home/rsp_test/mosaics/sentinel/NDVI.tif'
```

## Merging ESA-landcover files

We have 3 rasters of [ESA World Cover landcover](https://esa-worldcover.org/). We will use them as a target values in a training process.
```
>>> lcs = glob('/home/rsp_test/landcover/*.tif')
>>> print(lcs)
['/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N60E075_Map.tif',
 '/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N63E072_Map.tif',
 '/home/rsp_test/landcover/ESA_WorldCover_10m_2020_v100_N63E075_Map.tif']
```
We need to merge and clip them. To use Sentinel-2 and ESA landcover data together, we need to bring Landcover data to same resolution and projection as Sentinel-2 data. So we use one of Sentinel mosaic bands as a reference file using `reference_raster` argument.

```
>>> mosaic_landcover = rsp.mosaic(lcs, '/home/rsp_test/mosaics/landcover/', clipper = border, reference_raster = '/home/rsp_test/mosaics/sentinel/B1.tif', nodata = -1)
Processing completed
>>> print(mosaic_landcover)
['/home/rsp_test/mosaics/landcover/ESA_WorldCover_10m_2020_v100_N60E075_Map_mosaic.tif']
```
The function returns list of band mosaics.

## Cutting data to tiles


The goal of this tutorial is to create a model that predict land cover classes (y data) based on Sentinel-2 imagery (x data). We will use Convolutional Neural Network (CNN) for this task. CNN takes tiles of same size as input and process them one by one or unite them into mini-batches.

Here we define x data (that will be used by CNN as input training data) and y data (that will be used as target value).
```
>>> x = mosaic_sentinel
>>> y = mosaic_landcover[0]
```
We will cut Sentinel (x) and landcover (y) data to 256x256 px tiles (`tile_size = 256`). To have lower bias we will random shuffe tiles (`shuffle = True`). To evaluate model performance on a data that was not used in model training we will split data into train, validation and test subsets in proportion 3 to 1 to 1 (`split = [3, 1, 1]`).

The function returns list of x datasets (x train, x validation and x test), list of y datasets, list of tile coordinates, list of random shuffled samples, number of classes in data, class values, x and y nodata values. If you save the tiles into files, this data is also saved as files metadata.
```
>>> x_i, y_i, tiles, samples, classification, num_classes, classes, x_nodata, y_nodata = rsp.segmentation.generate_tiles(x, y, num_classes = 11, tile_size = 256, shuffle = True, split = [3, 1, 1], nodata = -1)
```
There are 3000 tiles in train set and 1000 tiles in both validation and test sets.
```
>>> x_train = x_i[0]
>>> print(x_train.shape)
(3000, 256, 256, 12)
>>> x_val = x_i[1]
>>> print(x_val.shape)
(1000, 256, 256, 12)
>>> x_test = x_i[2]
>>> print(x_test.shape)
(1000, 256, 256, 12)
>>> y_train = y_i[0]
>>> print(y_train.shape)
(3000, 256, 256, 11)
>>> y_val = y_i[1]
>>> print(y_val.shape)
(1000, 256, 256, 11)
>>> y_test = y_i[2]
>>> print(y_test.shape)
(1000, 256, 256, 11)
```
Samples are shuffled, task is classification.
```
>>> print(len(tiles))
5000
>>> print(samples[:5])
[1876, 684, 25, 7916, 1347]
>>> print(classification)
True
```
There are 11 classes in the data, their values are [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].
```
>>> print(num_classes)
11
>>> print(classes)
[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
>>> print(x_nodata)
0
>>> print(y_nodata)
0
```



## Training UperNet CNN

Here we are training UperNet CNN that predicts landcover class based on sentinel imagery.

![Vision transformer for sematnic segmentation of satellite imagery](https://pub.mdpi-res.com/remotesensing/remotesensing-13-03585/article_deploy/html/images/remotesensing-13-03585-ag.png?1631173514)

We will use the UperNet model architecture (`model == 'UperNet'`) with ConvNeXTV2 backbone (`backbone = 'ConvNeXTV2'`). Model will be saved to `/home/rsp_test/model/upernet.ckpt` and could be used later for testing and prediction. Model will be trained for 10 epochs (`epochs = 10`) with batch size of 32 tiles (`batch_size = 32`). Other parameters are defined from `generate_tiles` function output.
```
>>> model = rsp.segmentation.train(x_train, y_train, x_val, y_val, model = 'UperNet', backbone = 'ConvNeXTV2', model_file = '/home/rsp_test/model/upernet.ckpt', epochs = 10, batch_size = 32, classification = classification, num_classes = num_classes, x_nodata = x_nodata, y_nodata = y_nodata)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type                           | Params
-----------------------------------------------------------
0 | model   | UperNetForSemanticSegmentation | 59.8 M
1 | loss_fn | CrossEntropyLoss               | 0     
-----------------------------------------------------------
59.8 M    Trainable params
0         Non-trainable params
59.8 M    Total params
239.395   Total estimated model params size (MB)
Epoch 9: 100% #############################################
223/223 [1:56:20<00:00, 31.30s/it, v_num=54, train_loss_step=0.326, train_acc_step=0.871, train_auroc_step=0.796, train_iou_step=0.655,
val_loss_step=0.324, val_acc_step=0.869, val_auroc_step=0.620, val_iou_step=0.678,
val_loss_epoch=0.334, val_acc_epoch=0.807, val_auroc_epoch=0.795, val_iou_epoch=0.688,
train_loss_epoch=0.349, train_acc_epoch=0.842, train_auroc_epoch=0.797, train_iou_epoch=0.648]
`Trainer.fit` stopped: `max_epochs=10` reached.
```
Then we need to test model performance on test data.
```
>>> rsp.segmentation.test(x_test, y_test, model = model, batch_size = 32)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test_acc_epoch       │    0.8231202960014343     │
│     test_auroc_epoch      │    0.7588028311729431     │
│      test_iou_epoch       │    0.69323649406433105    │
│      test_loss_epoch      │    0.40799811482429504    │
│   test_precision_epoch    │    0.8231202960014343     │
│     test_recall_epoch     │    0.8231202960014343     │
└───────────────────────────┴───────────────────────────┘
```

## Mapping predictions

When we finished training a model, we can use it to create a landcover map based on its predictions. We need to define input data (it is out x train, validation and test datasets), tiles and samples generated by `generate_tiles` function, reference raster to get transform and CRS from (it is raster with our landcover data), model that will be used for predictions (it is out UperNet model) and a path where to write output map.
```
>>> y_reference = mosaic_landcover[0]
>>> rsp.segmentation.generate_map([x_train, x_val, x_test], y_reference, model, output_map, tiles = tiles, samples = samples, classes = classes)
Predicting: 100% #################### 372/372 [32:16, 1.6s/it]
```