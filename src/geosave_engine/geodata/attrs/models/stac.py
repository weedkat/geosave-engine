"""Per-acquisition STAC metadata carried with a raster."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime as dt
from typing import Annotated, Any, ClassVar, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from geosave_engine.geodata.attrs.model import AttrsModel
from geosave_engine.geodata.attrs.validate import parse_collection_text


class StacItem(BaseModel):
    """One STAC item as it was loaded.

    Args:
        id: Item ID.
        datetime: The item's own instant, naive UTC.
        properties: Item properties captured, keyed as STAC publishes them.
        assets: Asset name mapped to that asset's captured fields.

    Examples:
        >>> row.properties["eo:cloud_cover"]
        4.2
    """

    model_config = ConfigDict(frozen=True)

    id: str
    datetime: dt
    properties: dict[str, Any] = Field(default_factory=dict)
    assets: dict[str, dict[str, Any]] = Field(default_factory=dict)


class StacMetadata(AttrsModel):
    """Metadata of the STAC items a raster was loaded from.

    One row per item, each keeping its own identity and instant rather than an
    index into the loaded time axis, so a join accumulates provenance instead
    of realigning it.

    Args:
        stac_items: One row per item, in the order loaded.
        stac_groupby: How the items were grouped onto the time axis, as
            `odc.stac.load` was asked to group them.

    Examples:
        >>> ds.gs.attrs.root.get(StacMetadata).properties()
        ('eo:cloud_cover', 'platform')
    """

    NAME: ClassVar[str] = "stac"

    stac_items: Annotated[
        tuple[StacItem, ...] | None, BeforeValidator(parse_collection_text)
    ] = None
    stac_groupby: str | None = None

    def properties(self) -> tuple[str, ...]:
        """Name the item properties the rows carry.

        Returns:
            Property names across every row, sorted.
        """
        rows = self.stac_items or ()
        return tuple(sorted({key for row in rows for key in row.properties}))

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge STAC metadata: union `stac_items`, drop a mixed grouping.

        Items are provenance, so a joined result was loaded from all of them. A
        grouping the models do not share describes neither, so it is dropped
        rather than refused.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            Metadata carrying every model's items once, the first occurrence
            of an item id winning, and the attr keys it could not keep.

        Raises:
            ValueError: `models` is empty.
        """
        grouped, dropped = super().merge(models)
        items: dict[str, StacItem] = {}
        # The base merge has already refused any model that is not this model.
        for model in models:
            if not isinstance(model, cls):
                continue
            for item in model.stac_items or ():
                items.setdefault(item.id, item)
        combined = cls(
            stac_items=tuple(items.values()),
            stac_groupby=grouped.stac_groupby,
        )
        # Items accumulate rather than drop, so differing ones are not a loss.
        return combined, dropped - set(cls.field_keys["stac_items"])
