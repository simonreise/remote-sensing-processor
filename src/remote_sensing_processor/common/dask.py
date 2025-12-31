"""Dask functions. Now are not used."""

from typing import Any

import multiprocessing

from dask.distributed import LocalCluster


# TODO: Add Dask cluster support.
def get_client(cluster: Any) -> tuple[Any, Any]:
    """Function that gets dask client from a cluster, or creates a local cluster."""
    if cluster is None:
        cluster = get_local()
    client = cluster.get_client()
    return cluster, client


def get_local() -> LocalCluster:
    """Function that creates local cluster."""
    cluster = LocalCluster(memory_limit=None)
    cluster.adapt(minimum=1, maximum=multiprocessing.cpu_count())
    return cluster
