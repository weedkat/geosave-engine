from datetime import datetime
from dataclasses import replace

import pytest

from geosave_engine.geodata.stac.query import (
    StacQuery,
    Sentinel2L2AQuery,
    BaseQuery,
)


class TestStacQuery:
    """Test StacQuery dataclass."""

    def test_basic_initialization(self):
        """Query initializes with collections."""
        query = StacQuery(collections=["test-collection"])
        assert query.collections == ["test-collection"]
        assert query.bbox is None
        assert query.intersects is None

    def test_to_search_params_basic(self):
        """Convert query to search params dict."""
        query = StacQuery(collections=["col1"])
        params = query.to_search_params()
        
        assert params["collections"] == ["col1"]
        assert "bbox" not in params  # None values filtered out
        assert "intersects" not in params

    def test_to_search_params_with_bbox(self):
        """Bbox included when set."""
        query = StacQuery(
            collections=["col1"],
            bbox=(0, 0, 1, 1)
        )
        params = query.to_search_params()
        
        assert params["bbox"] == (0, 0, 1, 1)

    def test_to_search_params_with_datetime_string(self):
        """Datetime string passed through."""
        query = StacQuery(
            collections=["col1"],
            datetime="2023-01-01/2023-12-31"
        )
        params = query.to_search_params()
        
        assert params["datetime"] == "2023-01-01/2023-12-31"

    def test_to_search_params_with_datetime_object(self):
        """Datetime object passed through."""
        dt = datetime(2023, 1, 1)
        query = StacQuery(
            collections=["col1"],
            datetime=dt
        )
        params = query.to_search_params()
        
        assert params["datetime"] == dt

    def test_to_search_params_filter_sets_filter_lang(self):
        """Filter present → filter_lang set to cql2-json."""
        query = StacQuery(
            collections=["col1"],
            filter={"eo:cloud_cover": {"lte": 10}}
        )
        params = query.to_search_params()
        
        assert params["filter"] == {"eo:cloud_cover": {"lte": 10}}
        assert params["filter_lang"] == "cql2-json"

    def test_to_search_params_no_filter_no_filter_lang(self):
        """No filter → filter_lang not in params."""
        query = StacQuery(collections=["col1"])
        params = query.to_search_params()
        
        assert "filter_lang" not in params

    def test_frozen_dataclass(self):
        """Query is immutable."""
        query = StacQuery(collections=["col1"])
        
        with pytest.raises(Exception):  # FrozenInstanceError
            query.collections = ["col2"]

    def test_with_filter_new_filter(self):
        """with_filter adds filter when none exists."""
        query = StacQuery(collections=["col1"])
        new_expr = {"eo:cloud_cover": {"lte": 20}}
        
        filtered = query.with_filter(new_expr)
        
        assert filtered.filter == new_expr
        assert query.filter is None  # Original unchanged

    def test_with_filter_merge_existing(self):
        """with_filter merges with existing filter."""
        existing_filter = {"eo:cloud_cover": {"lte": 30}}
        query = StacQuery(
            collections=["col1"],
            filter=existing_filter
        )
        new_expr = {"platform": {"eq": "sentinel-2"}}
        
        filtered = query.with_filter(new_expr)
        
        # Should merge with AND logic
        assert filtered.filter is not None
        assert query.filter == existing_filter  # Original unchanged

    def test_with_filter_chainable(self):
        """with_filter returns same type for chaining."""
        query = StacQuery(collections=["col1"])
        
        result = query.with_filter({"a": 1}).with_filter({"b": 2})
        
        assert isinstance(result, StacQuery)
        assert result.filter is not None

    def test_multiple_collections(self):
        """Query accepts multiple collections."""
        query = StacQuery(collections=["col1", "col2", "col3"])
        
        assert query.collections == ["col1", "col2", "col3"]

    def test_ids_parameter(self):
        """Query with specific item IDs."""
        query = StacQuery(
            collections=["col1"],
            ids=["item1", "item2"]
        )
        params = query.to_search_params()
        
        assert params["ids"] == ["item1", "item2"]

    def test_limit_and_max_items(self):
        """Query with limit and max_items."""
        query = StacQuery(
            collections=["col1"],
            limit=100,
            max_items=1000
        )
        params = query.to_search_params()
        
        assert params["limit"] == 100
        assert params["max_items"] == 1000

    def test_sortby_string(self):
        """Sortby as string."""
        query = StacQuery(
            collections=["col1"],
            sortby="properties.datetime"
        )
        params = query.to_search_params()
        
        assert params["sortby"] == "properties.datetime"

    def test_sortby_list(self):
        """Sortby as list."""
        query = StacQuery(
            collections=["col1"],
            sortby=["+datetime", "-platform"]
        )
        params = query.to_search_params()
        
        assert params["sortby"] == ["+datetime", "-platform"]

    def test_implements_base_query_protocol(self):
        """StacQuery implements BaseQuery protocol."""
        query = StacQuery(collections=["col1"])
        
        assert isinstance(query, BaseQuery)
        assert hasattr(query, "to_search_params")

    def test_invalid_bbox_validation(self):
        """Invalid bbox raises error."""
        with pytest.raises(ValueError):  # Validation error
            StacQuery(
                collections=["col1"],
                bbox=(0, 50, 1, 40)  # Invalid: min_lat > max_lat
            )


class TestSentinel2L2AQuery:
    """Test Sentinel2L2AQuery specialized query."""

    def test_default_collection(self):
        """Default collection is sentinel-2-l2a."""
        query = Sentinel2L2AQuery()
        
        assert query.collections == ["sentinel-2-l2a"]

    def test_can_override_collections(self):
        """Can override default collection."""
        query = Sentinel2L2AQuery(collections=["custom-collection"])
        
        assert query.collections == ["custom-collection"]

    def test_max_cloud_cover_adds_filter(self):
        """max_cloud_cover adds cloud cover filter."""
        query = Sentinel2L2AQuery()
        
        filtered = query.max_cloud_cover(20)
        
        assert filtered.filter is not None
        params = filtered.to_search_params()
        assert params["filter_lang"] == "cql2-json"

    def test_max_cloud_cover_chainable(self):
        """max_cloud_cover returns same type."""
        query = Sentinel2L2AQuery()
        
        result = query.max_cloud_cover(20)
        
        assert isinstance(result, Sentinel2L2AQuery)

    def test_max_cloud_cover_with_existing_filter(self):
        """max_cloud_cover merges with existing filters."""
        query = Sentinel2L2AQuery(
            bbox=(0, 0, 1, 1)
        )
        
        filtered = query.max_cloud_cover(15)
        
        assert filtered.bbox == (0, 0, 1, 1)
        assert filtered.filter is not None

    def test_full_sentinel2_query(self):
        """Complete Sentinel2 query workflow."""
        query = (
            Sentinel2L2AQuery(bbox=(0, 0, 10, 10))
            .max_cloud_cover(25)
        )
        
        params = query.to_search_params()
        
        assert params["collections"] == ["sentinel-2-l2a"]
        assert params["bbox"] == (0, 0, 10, 10)
        assert params["filter_lang"] == "cql2-json"
