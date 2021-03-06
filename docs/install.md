# Installation

RSP needs python >= 3.7 to run and depends on several python libraries.

These libraries are:
- Numpy
- GDAL
- Rasterio
- Pyproj
- Shapely
- Fiona
- Geopandas
- Tensorflow >= 2.3

Most of them are installed when you install RSP, but you can have problems when installing GDAL, Rasterio and Fiona with pip. So it is recommended to install them manually.

Also you need a Sen2Cor to be installed to process Sentinel-2 data.

## PyPI

### Windows

Before installing RSP in Windows you have to download and install binary wheels for [GDAL](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal), [Rasterio](http://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio) and [Fiona](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona) from Christoph Gohlke's website.

Then install downloaded wheels.
```
pip install GDAL-3.4.3-cp39-cp39-win_amd64.whl
pip install rasterio-1.2.10-cp39-cp39-win_amd64.whl
pip install Fiona-1.8.21-cp39-cp39-win_amd64.whl
```
Now you can install RSP.
```
pip install remote-sensing-processor
```

### Linux
In Linux you need to install GDAL from apt manually.
```
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install python-numpy gdal-bin libgdal-dev
```
Now you can install RSP.
```
pip install remote-sensing-processor
```

### OS X

In OS X you need to install GDAL manually.
```
brew install gdal
```
Now you can install RSP.
```
pip install remote-sensing-processor
```

## Conda

Installing RSP in conda is much more easy.
```
conda install -c moskovchenkomike remote-sensing-processor
```

## Installing Sen2Cor

Sen2Cor is needed for Sentinel-2 atmospheric correction and conversion from L1 to L2. You should install it via SNAP plugin installer. [Here](http://wiki.awf.forst.uni-goettingen.de/wiki/index.php/Installation_of_SNAP) is the instruction how you can do it.

If you don't want to install SNAP, you can download and manually install [Sen2Cor 2.8](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor_v2-8/) to `%HOME%/.snap/auxdata/`.