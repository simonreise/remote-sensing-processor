import urllib.request
import pathlib
import remote_sensing_processor as rsp

archives = ["LM05_L1TP_161023_19930803_20211018_02_T2.tar", "LT05_L1TP_162023_20110812_20200820_02_T1.tar",
            "LT05_L1TP_160023_20110814_20200820_02_T1.tar", "LE07_L1TP_159023_20210826_20210921_02_T1.tar"]
archives = [str(pathlib.Path(__file__).parent.resolve()) + r'/' + a for a in archives]
urllib.request.urlretrieve("https://onedrive.live.com/download?cid=C9974FFDBF7F1C3A&resid=c9974ffdbf7f1c3a%2123147&authkey=AHz30c4JseIx7yI", archives[0])
urllib.request.urlretrieve("https://onedrive.live.com/download?cid=C9974FFDBF7F1C3A&resid=c9974ffdbf7f1c3a%2123149&authkey=AEays9RWJB_UleI", archives[1])
urllib.request.urlretrieve("https://onedrive.live.com/download?cid=C9974FFDBF7F1C3A&resid=c9974ffdbf7f1c3a%2123150&authkey=AGmTZ328PywOU4o", archives[2])
urllib.request.urlretrieve("https://onedrive.live.com/download?cid=C9974FFDBF7F1C3A&resid=c9974ffdbf7f1c3a%2123148&authkey=ADgB4Pw98Wfi580", archives[3])
landsats = rsp.landsat(archives)
assert landsats != None
clipper = str(pathlib.Path(__file__).parent.resolve()) + r'/' + 'roi.gpkg'
merged = rsp.mosaic(landsats, str(pathlib.Path(__file__).parent.resolve()), fill_nodata = True, clipper = clipper)
assert merged != None
ndvi = rsp.normalized_difference('NDVI', b1 = str(pathlib.Path(__file__).parent.resolve()) + '/B5.tif', b2 = str(pathlib.Path(__file__).parent.resolve()) + '/B4.tif')
assert ndvi != None