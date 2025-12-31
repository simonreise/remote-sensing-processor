"""Loading Sentinel 2."""

import planetary_computer
import pystac_client
import pytest

from pystac import Item
from pystac.extensions import eo


@pytest.fixture(scope="session")
def load_sentinel2_india() -> Item:
    """Load Sentinel 2 at India."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    india = [77.11, 28.59]
    time_of_interest = "2024-06-01/2024-08-31"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": india},
        datetime=time_of_interest,
    )
    items = search.item_collection()
    return min(items, key=lambda item: eo.EOExtension.ext(item).cloud_cover)
