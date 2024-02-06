# Installation

RSP needs python >= 3.8 to run and depends on several python libraries.

These libraries are:
- Python >= 3.8
- Numpy
- Xarray
- Dask
- Zarr
- GDAL
- Rasterio
- Rioxarray
- Pyproj
- Geopandas
- Scikit-learn
- Scikit-image
- XGBoost
- Pytorch >= 2.0
- Torchvision >= 0.10
- Lightning
- Transformers

Most of them are installed when you install RSP, but you can have problems when installing GDAL, Rasterio and Fiona with pip on Windows.

**If you have a GPU that supports CUDA we strongly recommend you to install Pytorch version that is built with CUDA support before installing RSP. You can find out how to do it on [Pytorch official site](https://pytorch.org/get-started/locally/).**

Also you need a Sen2Cor to be installed to process Sentinel-2 data.

## PyPI

You can install RSP via PIP.

You may need to install [Pytorch](https://pytorch.org/get-started/locally/) with CUDA support first.
```
pip install remote-sensing-processor
```

### Windows

If you run into problems when PIP is trying to install GDAL, Rasterio or Fiona, you can download binary wheels from [Christoph Gohlke's github](https://github.com/cgohlke/geospatial-wheels).

Then install downloaded wheels.
```
pip install GDAL-3.8.2-cp311-cp311-win_amd64.whl
pip install rasterio-1.3.9-cp311-cp311-win_amd64.whl
pip install Fiona-1.9.5-cp311-cp311-win_amd64.whl
```
Now you can install RSP.


### Linux
In Linux you may need to install GDAL from apt manually.
```
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install python3-numpy gdal-bin libgdal-dev
```
Now you can install RSP.

### OS X

In OS X you may need to install GDAL manually.
```
brew install gdal
```

## Conda

You can install RSP via Conda.

We recommend you to add [conda-forge](https://conda-forge.org/) channel to your channel list because some of the requirements are not available on main channel. You may also need to add `pytorch` and `nvidia` channels to install Pytorch that supports CUDA.
```
conda install -c moskovchenkomike remote-sensing-processor
```

## Installing Sen2Cor

Sen2Cor is needed for Sentinel-2 atmospheric correction and conversion from L1 to L2.

Required version is 2.11 for Windows and Linux and 2.9 for Mac OS.

You should install it via SNAP plugin installer. [Here](http://wiki.awf.forst.uni-goettingen.de/wiki/index.php/Installation_of_SNAP) is the instruction how you can do it.

If you don't want to install SNAP, you can download and manually install [Sen2Cor 2.11](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-11/) or [Sen2Cor 2.9](https://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-9/) to `%HOME%/.snap/auxdata/`.