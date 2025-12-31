"""Loading ESA Worldcover."""

import planetary_computer
import pystac_client
import pytest

from pystac import Item


@pytest.fixture(scope="session")
def load_lulc_roma() -> Item:
    """Load ESA Worldcover at Roma."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    roma = [12.45, 41.87]
    search = catalog.search(collections=["esa-worldcover"], intersects={"type": "Point", "coordinates": roma})
    items = list(search.items())
    item = items[0]
    item.assets = {"map": item.assets["map"]}
    for i in range(len(item.assets["map"].extra_fields["classification:classes"])):
        if "name" not in item.assets["map"].extra_fields["classification:classes"][i]:
            item.assets["map"].extra_fields["classification:classes"][i]["name"] = (
                item.assets["map"]
                .extra_fields["classification:classes"][i]["description"]
                .replace(" ", "-")
                .replace("/", "-")
            )
    return item
