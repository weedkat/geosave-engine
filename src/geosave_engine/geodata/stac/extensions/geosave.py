"""GeoSave extension: the fields GeoSave adds to a STAC Item."""

from __future__ import annotations

from typing import Literal

import pystac
from pystac.extensions.base import ExtensionManagementMixin, PropertiesExtension

SCHEMA_URI = (
    "https://raw.githubusercontent.com/weedkat/geosave-engine/main/"
    "schemas/geosave/v0.1.0/schema.json"
)

STACK_PROP = "geosave:stack"


class GeosaveExtension(PropertiesExtension, ExtensionManagementMixin[pystac.Item]):
    """GeoSave's own properties on one Item.

    Args:
        item: Item to read or write.

    Examples:
        >>> GeosaveExtension.ext(item, add_if_missing=True).stack = "s0"
        >>> GeosaveExtension.ext(item).stack
        's0'
    """

    name: Literal["geosave"] = "geosave"

    def __init__(self, item: pystac.Item) -> None:
        """Wrap the properties of one Item."""
        self.item = item
        self.properties = item.properties

    @property
    def stack(self) -> str | None:
        """Return the stack this Item was saved from, shared by its groups."""
        return self._get_property(STACK_PROP, str)

    @stack.setter
    def stack(self, value: str | None) -> None:
        self._set_property(STACK_PROP, value)

    @classmethod
    def get_schema_uri(cls) -> str:
        """Return the schema an Item declares when it uses these fields."""
        return SCHEMA_URI

    @classmethod
    def ext(cls, obj: pystac.Item, add_if_missing: bool = False) -> GeosaveExtension:
        """Extend an Item with GeoSave's properties.

        Args:
            obj: Item to extend.
            add_if_missing: Declare the schema on the Item when it is absent.

        Returns:
            The extension over the Item's properties.

        Raises:
            pystac.ExtensionNotImplemented: The Item does not declare the
                schema and `add_if_missing` is false.
        """
        cls.ensure_has_extension(obj, add_if_missing)
        return cls(obj)
