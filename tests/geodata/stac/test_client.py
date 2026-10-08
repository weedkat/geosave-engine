from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
import pytest

from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.errors import CollectionNotFoundError


def test_wrapping_a_client_preserves_its_transport():
    transport = StacApiIO(headers={"X-Catalog": "private"}, timeout=7)
    client = Client(id="local", description="Custom transport")
    client._stac_io = transport

    StacClient(client)

    assert client._stac_io is transport
    assert transport.session.headers["X-Catalog"] == "private"


def test_missing_collection_raises_a_lookup_error(monkeypatch):
    native = Client(id="local", description="Local catalog")
    monkeypatch.setattr(native, "get_collection", lambda _name: None)

    with pytest.raises(CollectionNotFoundError, match="missing"):
        StacClient(native).collection("missing")


@pytest.mark.parametrize("provider", ["planetary_computer", "cdse", "element84"])
def test_named_provider_uses_its_stac_client_constructor(monkeypatch, provider):
    expected = object()
    monkeypatch.setattr(
        StacClient,
        provider,
        classmethod(lambda cls: expected),
    )

    assert StacClient.open(provider) is expected
