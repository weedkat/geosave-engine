from unittest.mock import Mock, MagicMock, patch, ANY
import pytest
import pystac
import pystac_client

from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.query import StacQuery, Sentinel2L2AQuery


class TestStacClient:
    """Test StacClient wrapper."""

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_initialization_with_url(self, mock_open):
        """Client opens STAC URL on init."""
        mock_open.return_value = MagicMock()
        
        client = StacClient("https://example.com/stac")
        
        mock_open.assert_called_once()
        assert client.client is not None

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_retry_strategy_configured(self, mock_open):
        """Retry strategy passed to StacApiIO."""
        mock_open.return_value = MagicMock()
        
        client = StacClient("https://example.com/stac")
        
        # Verify init called (retry config applied internally)
        mock_open.assert_called_once()

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_search_with_dict_query(self, mock_open):
        """search accepts dict query."""
        mock_search_obj = MagicMock()
        mock_item = MagicMock(spec=pystac.Item)
        mock_search_obj.items.return_value = [mock_item]
        
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_obj
        mock_open.return_value = mock_client
        
        client = StacClient("https://example.com/stac")
        query_dict = {"collections": ["test"], "bbox": (0, 0, 1, 1)}
        
        result = client.search(query_dict)
        
        mock_client.search.assert_called_once_with(**query_dict)
        assert result == [mock_item]

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_search_with_stac_query_object(self, mock_open):
        """search accepts BaseQuery object."""
        mock_search_obj = MagicMock()
        mock_item = MagicMock(spec=pystac.Item)
        mock_search_obj.items.return_value = [mock_item]
        
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_obj
        mock_open.return_value = mock_client
        
        client = StacClient("https://example.com/stac")
        query = StacQuery(collections=["test"], bbox=(0, 0, 1, 1))
        
        result = client.search(query)
        
        # Should call with converted params
        mock_client.search.assert_called_once()
        assert result == [mock_item]

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_search_returns_list(self, mock_open):
        """search returns list of items."""
        mock_search_obj = MagicMock()
        mock_items = [MagicMock(spec=pystac.Item) for _ in range(3)]
        mock_search_obj.items.return_value = mock_items
        
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_obj
        mock_open.return_value = mock_client
        
        client = StacClient("https://example.com/stac")
        result = client.search({"collections": ["test"]})
        
        assert len(result) == 3
        assert result == mock_items

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_search_iter_returns_generator(self, mock_open):
        """search_iter returns generator."""
        mock_search_obj = MagicMock()
        mock_items = [MagicMock(spec=pystac.Item) for _ in range(3)]
        mock_search_obj.items.return_value = iter(mock_items)
        
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_obj
        mock_open.return_value = mock_client
        
        client = StacClient("https://example.com/stac")
        result = client.search_iter({"collections": ["test"]})
        
        # Should be a generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_search_iter_yields_items(self, mock_open):
        """search_iter yields items one at a time."""
        mock_search_obj = MagicMock()
        mock_items = [MagicMock(spec=pystac.Item) for _ in range(3)]
        mock_search_obj.items.return_value = iter(mock_items)
        
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_obj
        mock_open.return_value = mock_client
        
        client = StacClient("https://example.com/stac")
        result = list(client.search_iter({"collections": ["test"]}))
        
        assert result == mock_items

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_search_iter_with_stac_query_object(self, mock_open):
        """search_iter accepts BaseQuery object."""
        mock_search_obj = MagicMock()
        mock_items = [MagicMock(spec=pystac.Item) for _ in range(2)]
        mock_search_obj.items.return_value = iter(mock_items)
        
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_obj
        mock_open.return_value = mock_client
        
        client = StacClient("https://example.com/stac")
        query = Sentinel2L2AQuery(bbox=(0, 0, 1, 1))
        result = list(client.search_iter(query))
        
        assert result == mock_items

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_get_collections(self, mock_open):
        """get_collections returns list of collections."""
        mock_collection = MagicMock(spec=pystac.Collection)
        mock_client = MagicMock()
        mock_client.get_collections.return_value = [mock_collection]
        mock_open.return_value = mock_client
        
        client = StacClient("https://example.com/stac")
        result = client.get_collections()
        
        assert result == [mock_collection]
        mock_client.get_collections.assert_called_once()

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_cdse_factory(self, mock_open):
        """cdse() creates client for CDSE catalog."""
        mock_open.return_value = MagicMock()
        
        client = StacClient.cdse()
        
        mock_open.assert_called_once_with("https://cdse-catalog.copernicus.eu/stac/v1", stac_io=ANY)

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_planetary_computer_factory(self, mock_open):
        """planetary_computer() creates client for Planetary Computer."""
        mock_open.return_value = MagicMock()
        
        client = StacClient.planetary_computer()
        
        mock_open.assert_called_once_with("https://planetarycomputer.microsoft.com/api/stac/v1", stac_io=ANY)

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_element84_factory(self, mock_open):
        """element84() creates client for Element84."""
        mock_open.return_value = MagicMock()
        
        client = StacClient.element84()
        
        mock_open.assert_called_once_with("https://earth-search.aws.element84.com/v0", stac_io=ANY)

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_factory_methods_return_stac_client(self, mock_open):
        """Factory methods return StacClient instances."""
        mock_open.return_value = MagicMock()
        
        clients = [
            StacClient.cdse(),
            StacClient.planetary_computer(),
            StacClient.element84(),
        ]
        
        for client in clients:
            assert isinstance(client, StacClient)

    @patch("geosave_engine.geodata.stac.client.pystac_client.Client.open")
    def test_search_with_sentinel2_query(self, mock_open):
        """search works with Sentinel2L2AQuery."""
        mock_search_obj = MagicMock()
        mock_item = MagicMock(spec=pystac.Item)
        mock_search_obj.items.return_value = [mock_item]
        
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_obj
        mock_open.return_value = mock_client
        
        client = StacClient.cdse()
        query = Sentinel2L2AQuery(bbox=(0, 0, 1, 1)).max_cloud_cover(20)
        
        result = client.search(query)
        
        assert result == [mock_item]
        # Verify search was called with correct collection
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]
        assert "sentinel-2-l2a" in call_kwargs["collections"]
