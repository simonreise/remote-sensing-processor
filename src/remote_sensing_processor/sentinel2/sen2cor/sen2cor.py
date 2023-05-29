import os
import sys
from sys import platform
import pathlib
import subprocess
import shutil
import urllib.request
import zipfile
from glob import glob


"""#function for handling too long ways
def winapi_path(dos_path, encoding=None):
    path = os.path.abspath(dos_path)

    if path.startswith("\\\\"):
        path = "\\\\?\\UNC\\" + path[2:]
    else:
        path = "\\\\?\\" + path 

    return path  """

"""def check_install(ver):
    path = str(pathlib.Path(__file__).parents[0].joinpath(ver))
    if not os.path.exists(path):
        #finding and deleting old versions
        folders = glob(str(pathlib.Path(__file__).parents[0]) + '/Sen2Cor*')
        for folder in folders:
            os.remove(folder)
        #downloading and unpacking current sen2cor version
        if platform == "linux" or platform == "linux2":
            filename = pathlib.Path(__file__).parents[0].joinpath('Sen2Cor-02.10.01-Linux64.run')
            urllib.request.urlretrieve('https://step.esa.int/thirdparties/sen2cor/2.10.0/Sen2Cor-02.10.01-Linux64.run', filename)
            subprocess.run(['/bin/bash', str(filename)])
        elif platform == "darwin":
            filename = pathlib.Path(__file__).parents[0].joinpath('Sen2Cor-02.10.01-Darwin64.run')
            urllib.request.urlretrieve('https://step.esa.int/thirdparties/sen2cor/2.10.0/Sen2Cor-02.10.01-Darwin64.run', filename)
            subprocess.run(['/bin/bash', str(filename)])
        elif platform == "win32":
            filename = pathlib.Path(__file__).parents[0].joinpath('Sen2Cor-02.10.01-win64.zip')
            urllib.request.urlretrieve('https://step.esa.int/thirdparties/sen2cor/2.10.0/Sen2Cor-02.10.01-win64.zip', filename)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                targetpath = winapi_path(pathlib.Path(__file__).parents[0])
                zip_ref.extractall(targetpath)
        os.remove(filename)"""
        
        
def sen2correct(input_path):
    if platform == "linux" or platform == "linux2":
        ver = 'Sen2Cor-02.08.00-Linux64'
    elif platform == "darwin":
        ver = 'Sen2Cor-02.08.00-Darwin64'
    elif platform == "win32":
        ver = 'Sen2Cor-02.08.00-win64'
    #check_install(ver)
    my_env = os.environ.copy()
    #print(pathlib.Path.home().joinpath(r'.snap/auxdata/' + ver))
    my_env['PATH'] = str(pathlib.Path.home().joinpath(r'.snap/auxdata/' + ver + '/bin/')) + os.pathsep + str(pathlib.Path.home().joinpath(r'.snap/auxdata/' + ver)) + os.pathsep + my_env['PATH']
    my_env['SEN2COR_HOME'] = str(pathlib.Path.home().joinpath(r'.snap/auxdata/' + ver))
    my_env['SEN2COR_BIN'] = str(pathlib.Path.home().joinpath(r'.snap/auxdata/' + ver + '/lib/python2.7/site-packages/sen2cor'))
    my_env['LC_NUMERIC'] = 'C'
    my_env['GDAL_DATA'] = str(pathlib.Path.home().joinpath(r'.snap/auxdata/' + ver + '/share/gdal'))
    my_env['GDAL_DRIVER_PATH']= 'disable'
    try:
        if platform == "linux" or platform == "linux2":
            cmd = r'L2A_Process "'+ input_path + '" '
        elif platform == "darwin":
            cmd = r'L2A_Process "'+ input_path + '" '
        elif platform == "win32":
            cmd = r'L2A_Process.bat "'+ input_path + '" '
        result = subprocess.check_output(cmd, env=my_env, shell=True)
        shutil.rmtree(input_path)
        #print(path)
    except subprocess.CalledProcessError as e:
        print('Sen2Cor not working. Is it installed correctly?')
        print(e)
        sys.exit(1)