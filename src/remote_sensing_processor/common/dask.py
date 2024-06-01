import multiprocessing

from dask.distributed import LocalCluster


def get_client(cluster):
    if cluster is None:
        cluster = get_local()
    client = cluster.get_client()
    return client
        

def get_local():
    cluster = LocalCluster(memory_limit=None)
    cluster.adapt(minimum=1, maximum=multiprocessing.cpu_count())
    return cluster