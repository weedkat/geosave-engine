from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO

from geosave_engine.geodata.stac import StacClient


def test_wrapping_a_client_preserves_its_transport():
    transport = StacApiIO(headers={"X-Catalog": "private"}, timeout=7)
    client = Client(id="local", description="Custom transport")
    client._stac_io = transport

    StacClient(client)

    assert client._stac_io is transport
    assert transport.session.headers["X-Catalog"] == "private"
