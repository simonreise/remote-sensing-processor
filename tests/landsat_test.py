import urllib.request
import pathlib
import remote_sensing_processor as rsp


urllib.request.urlretrieve("https://drive.google.com/uc?export=download&id=1HO8aHfi8AJ5vfa9KJXBLGsW2ev3i0cZz", "LM05_L1TP_161023_19930803_20211018_02_T2.tar")
urllib.request.urlretrieve("https://drive.google.com/uc?export=download&id=1_R8qUu6JMzSEHtnAtqAs-DLDeWJj9EX5", "LT05_L1TP_162023_20110812_20200820_02_T1.tar")
urllib.request.urlretrieve("https://drive.google.com/uc?export=download&id=1egz0ZRqAZKIcZ7jWQnbPDkZ_WoCALSh6", "LT05_L1TP_160023_20110814_20200820_02_T1.tar")
urllib.request.urlretrieve("https://drive.google.com/uc?export=download&id=100lt-ssXcDtrpnE2nZIvSWXXomZK3YJ2", "LE07_L1TP_159023_20210826_20210921_02_T1.tar")
archives = ["LM05_L1TP_161023_19930803_20211018_02_T2.tar", "LT05_L1TP_162023_20110812_20200820_02_T1.tar",
            "LT05_L1TP_160023_20110814_20200820_02_T1.tar", "LE07_L1TP_159023_20210826_20210921_02_T1.tar"]
archives = [str(pathlib.Path(__file__).parent.resolve()) + r'\\' + a for a in archives]
landsats = rsp.landsat(archives)
assert landsats != None
clipper = str(pathlib.Path(__file__).parent.resolve()) + r'\\' + 'roi.gpkg'
merged = rsp.mosaic(landsats, str(pathlib.Path(__file__).parent.resolve()), fill_nodata = True, clipper = clipper)
assert merged != None
ndvi = rsp.normalized_difference('NDVI', merged)
assert ndvi != None