"""Sen2Cor wrapper."""

import os
import pathlib
import shutil
import subprocess
from sys import platform


def sen2correct(input_path: pathlib.Path) -> None:
    """Run Sen2Cor."""
    if platform == "linux" or platform == "linux2":
        ver = "Sen2Cor-02.12.03-Linux64"
    elif platform == "darwin":
        ver = "Sen2Cor-02.09.00-Darwin64"
    elif platform == "win32":
        ver = "Sen2Cor-02.12.03-win64"
    else:
        raise OSError("Unknown OS")

    # check_install(ver)
    my_env = os.environ.copy()
    # print(pathlib.Path.home().joinpath(r'.snap/auxdata/' + ver))
    my_env["PATH"] = (
        str(pathlib.Path.home().joinpath(r".snap/auxdata/" + ver + "/bin/"))
        + os.pathsep
        + str(pathlib.Path.home().joinpath(r".snap/auxdata/" + ver))
        + os.pathsep
        + my_env["PATH"]
    )
    my_env["SEN2COR_HOME"] = str(pathlib.Path.home().joinpath(r".snap/auxdata/" + ver))
    my_env["SEN2COR_BIN"] = str(
        pathlib.Path.home().joinpath(r".snap/auxdata/" + ver + "/lib/python2.7/site-packages/sen2cor"),
    )
    my_env["LC_NUMERIC"] = "C"
    my_env["GDAL_DATA"] = str(pathlib.Path.home().joinpath(r".snap/auxdata/" + ver + "/share/gdal"))
    my_env["GDAL_DRIVER_PATH"] = "disable"
    try:
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            cmd = r'L2A_Process "' + str(input_path.resolve()) + '" '
        elif platform == "win32":
            cmd = r'L2A_Process.bat "' + str(input_path.resolve()) + '" '
        else:
            raise OSError("Unknown OS")

        result = subprocess.check_output(cmd, env=my_env, shell=True)  # noqa: S602
        if "Application terminated with at least one error" in result.decode():
            raise RuntimeError(result.decode())
        shutil.rmtree(input_path)
        # print(path)
    except subprocess.CalledProcessError as e:
        print(e)
        raise OSError("Sen2Cor not working. Is it installed correctly?") from e
