"""Zarr extension: the store metadata needed to open a saved hierarchy."""

from __future__ import annotations

from typing import Literal

import pystac
import xarray as xr
from pystac.extensions.base import ExtensionManagementMixin, PropertiesExtension

SCHEMA_URI = "https://stac-extensions.github.io/zarr/v1.1.0/schema.json"

ZARR_FORMAT_PROP = "zarr:zarr_format"
NODE_TYPE_PROP = "zarr:node_type"
CONSOLIDATED_PROP = "zarr:consolidated"


class ZarrExtension(PropertiesExtension, ExtensionManagementMixin[pystac.Item]):
    """Zarr store metadata on one asset.

    PySTAC ships no class for the Zarr extension, so this one follows the
    shape of its own: build it with `ext`, write with `apply`.

    Args:
        asset: Asset pointing at a Zarr store.

    Examples:
        >>> zarr = ZarrExtension.ext(asset, add_if_missing=True)
        >>> zarr.apply(zarr_format=3, node_type="group", consolidated=False)
        >>> zarr.zarr_format
        3
    """

    name: Literal["zarr"] = "zarr"

    def __init__(self, asset: pystac.Asset) -> None:
        """Wrap the fields of one asset."""
        self.asset = asset
        self.properties = asset.extra_fields

    def apply(self, *, zarr_format: int, node_type: str, consolidated: bool) -> None:
        """State how the store is laid out.

        Args:
            zarr_format: Zarr specification version the store follows, 2 or 3.
            node_type: What the href points at, `"group"` or `"array"`.
            consolidated: Whether the store carries consolidated metadata.
        """
        self.zarr_format = zarr_format
        self.node_type = node_type
        self.consolidated = consolidated

    @property
    def zarr_format(self) -> int | None:
        """Return the Zarr specification version the store follows."""
        return self._get_property(ZARR_FORMAT_PROP, int)

    @zarr_format.setter
    def zarr_format(self, value: int | None) -> None:
        self._set_property(ZARR_FORMAT_PROP, value)

    @property
    def node_type(self) -> str | None:
        """Return whether the href points at a group or an array."""
        return self._get_property(NODE_TYPE_PROP, str)

    @node_type.setter
    def node_type(self, value: str | None) -> None:
        self._set_property(NODE_TYPE_PROP, value)

    @property
    def consolidated(self) -> bool | None:
        """Return whether the store carries consolidated metadata."""
        return self._get_property(CONSOLIDATED_PROP, bool)

    @consolidated.setter
    def consolidated(self, value: bool | None) -> None:
        self._set_property(CONSOLIDATED_PROP, value)

    @classmethod
    def get_schema_uri(cls) -> str:
        """Return the schema an Item declares when an asset uses these fields."""
        return SCHEMA_URI

    @classmethod
    def ext(cls, obj: pystac.Asset, add_if_missing: bool = False) -> ZarrExtension:
        """Extend an asset with the Zarr fields.

        Args:
            obj: Asset already added to its Item.
            add_if_missing: Declare the schema on the Item when it is absent.

        Returns:
            The extension over the asset's fields.

        Raises:
            pystac.ExtensionNotImplemented: The Item does not declare the
                schema and `add_if_missing` is false.
        """
        cls.ensure_owner_has_extension(obj, add_if_missing)
        return cls(obj)


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write the layout of the Zarr store a raster was opened from.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `read_raster` opens it. One not opened from a Zarr
            store writes nothing.

    Examples:
        >>> write(asset, read_raster("samples/forest.zarr"))
        >>> ZarrExtension.ext(asset).node_type
        'group'
    """
    store = raster.encoding.get("zarr")
    if store is None:
        return
    ZarrExtension.ext(asset, add_if_missing=True).apply(
        zarr_format=store["zarr_format"],
        node_type=store["node_type"],
        consolidated=store["consolidated"],
    )
