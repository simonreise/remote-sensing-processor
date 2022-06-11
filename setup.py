import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
	name = 'remote-sensing-processor',
	version = '0.0.1',
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
    setup_requires = ['cython', 'numpy'],
    install_requires = [ 'numpy', 'h5py', 'tensorflow>=2.3', 'scikit-image', 'rasterio', 'pyproj', 'geopandas'],
    python_requires = ">=3.7",
    include_package_data = True
)