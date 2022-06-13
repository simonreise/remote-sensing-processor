set "$py=0"
call:construct

for /f "delims=" %%a in ('%PYTHON% #.py ^| findstr "3.7"') do set "$py=3.7"
for /f "delims=" %%a in ('%PYTHON% #.py ^| findstr "3.8"') do set "$py=3.8"
for /f "delims=" %%a in ('%PYTHON% #.py ^| findstr "3.9"') do set "$py=3.9"
for /f "delims=" %%a in ('%PYTHON% #.py ^| findstr "3.10"') do set "$py=3.10"
del #.py
goto:%$py%

:3.7
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/GDAL-3.4.2-cp37-cp37m-win_amd64.whl"
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/rasterio-1.2.10-cp37-cp37m-win_amd64.whl"
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/Fiona-1.8.21-cp37-cp37m-win_amd64.whl"
goto end

:3.8
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/GDAL-3.4.3-cp38-cp38-win_amd64.whl"
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/rasterio-1.2.10-cp38-cp38-win_amd64.whl"
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/Fiona-1.8.21-cp38-cp38-win_amd64.whl"
goto end

:3.9
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/GDAL-3.4.3-cp39-cp39-win_amd64.whl"
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/rasterio-1.2.10-cp39-cp39-win_amd64.whl"
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/Fiona-1.8.21-cp39-cp39-win_amd64.whl"
goto end

:3.10
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/GDAL-3.4.3-cp310-cp310-win_amd64.whl"
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/rasterio-1.2.10-cp310-cp310-win_amd64.whl"
"%PYTHON%" -m pip install https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/W73491bhV-Cbsg/Fiona-1.8.21-cp310-cp310-win_amd64.whl"
goto end

:end
"%PYTHON%" setup.py install
if errorlevel 1 exit 1

:construct
echo import sys; print('{0[0]}.{0[1]}'.format(sys.version_info^)^) >#.py