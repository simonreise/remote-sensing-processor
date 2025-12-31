# Installation

RSP needs python >= 3.12 to run and depends on several python libraries.

Most of them are installed when you install RSP, but you can have problems when installing GDAL, Rasterio and Fiona with pip on Windows.

**If you have a GPU that supports CUDA we strongly recommend you to install Pytorch version that is built with CUDA support before installing RSP. You can find out how to do it on [Pytorch official site](https://pytorch.org/get-started/locally/).**

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
pip install GDAL-3.11.4-cp312-cp312-win_amd64.whl
pip install rasterio-1.4.3-cp312-cp312-win_amd64.whl
pip install Fiona-1.10.1-cp312-cp312-win_amd64.whl
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

We recommend you to add [conda-forge](https://conda-forge.org/) channel to your channel list because some of the requirements are not available on main channel.
```
conda install -c moskovchenkomike remote-sensing-processor
```