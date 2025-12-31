"""Loading Landsat."""

import planetary_computer
import pystac_client
import pytest

from pystac import Item
from pystac.extensions import eo


@pytest.fixture(scope="session")
def load_landsat8_brazil() -> Item:
    """Load Landsat 8 at Brazil."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    brazil = [-60.2, -3.31]
    time_of_interest = "2021-06-01/2021-08-31"
    search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects={"type": "Point", "coordinates": brazil},
        datetime=time_of_interest,
    )
    items = search.item_collection()
    return min(items, key=lambda item: eo.EOExtension.ext(item).cloud_cover)


@pytest.fixture(scope="session")
def load_landsat5_brazil() -> Item:
    """Load Landsat 5 at Brazil."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    brazil = [-60.2, -3.31]
    time_of_interest = "1998-06-01/1998-08-31"
    search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects={"type": "Point", "coordinates": brazil},
        datetime=time_of_interest,
    )
    items = search.item_collection()
    return min(items, key=lambda item: eo.EOExtension.ext(item).cloud_cover)


@pytest.fixture(scope="session")
def load_landsat1_brazil() -> list[Item]:
    """Load two Landsat 1 at Brazil."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    brazil = [-60.2, -3.31]
    time_of_interest = "1985-06-01/1985-08-31"
    search = catalog.search(
        collections=["landsat-c2-l1"],
        intersects={"type": "Point", "coordinates": brazil},
        datetime=time_of_interest,
    )
    # noinspection PyTypeChecker
    return search.item_collection()[0:2]


@pytest.fixture(scope="session")
def load_landsat_beijing() -> Item:
    """Load Landsat at China."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    beijing = [116.45, 39.91]
    time_of_interest = "2021-06-01/2021-08-31"
    search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects={"type": "Point", "coordinates": beijing},
        datetime=time_of_interest,
    )
    items = search.item_collection()
    return min(items, key=lambda item: eo.EOExtension.ext(item).cloud_cover)


@pytest.fixture(scope="session")
def load_landsat_beijing_match_hist() -> tuple[Item, Item]:
    """Load Landsat at China for histogram matching."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    beijing = [116.45, 39.91]

    time_of_interest = "2021-06-01/2021-08-31"
    search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects={"type": "Point", "coordinates": beijing},
        datetime=time_of_interest,
    )
    items = search.item_collection()
    item = min(items, key=lambda item: eo.EOExtension.ext(item).cloud_cover)
    item.assets = {"nir08": item.assets["nir08"]}
    item.id = "nir2021"

    time_of_interest = "2022-06-01/2022-08-31"
    search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects={"type": "Point", "coordinates": beijing},
        datetime=time_of_interest,
    )
    items = search.item_collection()
    item1 = min(items, key=lambda item: eo.EOExtension.ext(item).cloud_cover)
    item1.assets = {"nir08": item1.assets["nir08"]}
    item1.id = "nir2022"
    return item, item1
