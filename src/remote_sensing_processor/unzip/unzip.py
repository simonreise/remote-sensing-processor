import shutil

def unzip_sentinel(archive):
    path = archive.replace('.zip', '/')
    shutil.unpack_archive(archive, path)
    return path
    
def unzip_landsat(archive):
    if '.tar.gz' in archive:
        path = archive.replace('.tar.gz', '/')
    elif '.tar' in archive:
        path = archive.replace('.tar', '/')
    shutil.unpack_archive(archive, path)
    return path