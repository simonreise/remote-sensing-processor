import os
import shutil


def unzip_sentinel(archive):
    if os.path.isfile(archive) and '.zip' in archive:
        path = archive.replace('.zip', '/')
        if os.path.exists(path):
            shutil.rmtree(path)
        shutil.unpack_archive(archive, path)
    elif os.path.isdir(archive):
        path = archive
    else:
        raise ValueError(archive + ' is not a directory or zip archive')
    return path
    
def unzip_landsat(archive):
    if os.path.isfile(archive) and ('.tar.gz' in archive or '.tar' in archive):
        if '.tar.gz' in archive:
            path = archive.replace('.tar.gz', '/')
        elif '.tar' in archive:
            path = archive.replace('.tar', '/')
        if os.path.exists(path):
            shutil.rmtree(path)
        shutil.unpack_archive(archive, path)
    elif os.path.isdir(archive):
        path = archive
    else:
        raise ValueError(archive + ' is not a directory or tar/tar.gz archive')
    return path