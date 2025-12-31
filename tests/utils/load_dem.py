"""Loading Copernicus DEM."""

import planetary_computer
import pystac_client
import pytest

from pystac import Item


@pytest.fixture(scope="session")
def load_dem_grand_canyon() -> Item:
    """Loading Copernicus DEM at the Grand Canyon Area."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    grand_canyon = [-112.15, 36.05]
    search = catalog.search(collections=["cop-dem-glo-30"], intersects={"type": "Point", "coordinates": grand_canyon})
    items = list(search.items())
    item = items[0]
    item.assets = {"data": item.assets["data"]}
    return item
