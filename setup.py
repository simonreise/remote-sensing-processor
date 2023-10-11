import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("src/remote_sensing_processor/__init__.py") as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

setuptools.setup(
	name = 'remote-sensing-processor',
	version = version,
	author = 'Mikhail Moskovchenko',
	author_email = 'moskovchenkomike@gmail.com',
	description = 'RSP is a tool for geospatial raster data processing',
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	url = 'https://github.com/simonreise/remote-sensing-processor',
	project_urls = {
		'Bug Tracker': 'https://github.com/simonreise/remote-sensing-processor/issues',
        'Source': 'https://github.com/simonreise/remote-sensing-processor',
        'Documentation': 'https://remote-sensing-processor.readthedocs.io'
	},
	classifiers = [
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Operating System :: OS Independent',
		'Environment :: GPU :: NVIDIA CUDA',
		'Topic :: Scientific/Engineering :: GIS'
	],
    keywords = 'remote sensing, landsat, sentinel, gdal, rasterio',
	package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    setup_requires = ['cython', 'numpy>=1.17'],
    install_requires = ['numpy>=1.17', 'h5py', 'torch>=1.10', 'torchvision>=0.10', 'lightning', 'tensorboard', 'transformers', 'timm', 'scikit-learn', 'scikit-image', 'rasterio', 'pyproj', 'geopandas', 'albumentations'],
    python_requires = ">=3.8",
    include_package_data = True
)