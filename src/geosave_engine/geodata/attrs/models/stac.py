"""STAC source history carried with a raster."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime as dt
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from geosave_engine.geodata.attrs.model import AttrsModel, parse_collection_text


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

    Source records keep their own identity and instant rather than indexing
    the loaded time axis. Joins accumulate this history, and selecting pixels
    leaves it intact. Use `with_properties` for time-dependent values.

    Args:
        stac_items: Captured source records, in the order first encountered.
        stac_groupby: How the items were grouped onto the time axis, as
            `odc.stac.load` was asked to group them.

    Examples:
        >>> ds.gs.attrs.root.get(StacMetadata).properties()
        ('eo:cloud_cover', 'platform')
    """

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
    def merge(
        cls,
        models: Sequence[AttrsModel | None],
        *,
        conflicts: Literal["raise", "drop"] = "raise",
    ) -> tuple[Self, set[str]]:
        """Merge STAC metadata: union `stac_items`, drop a mixed grouping.

        Items are provenance, so a joined result was loaded from all of them. A
        grouping the models do not share describes neither, so it is dropped
        rather than refused.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.
            conflicts: Whether differing `MUST_AGREE` fields raise or drop.

        Returns:
            Metadata carrying every distinct source record once, in encounter
            order, and the attr keys it could not keep. Records sharing an ID
            but differing in timestamp, properties, or assets are retained.

        Raises:
            ValueError: `models` is empty.
        """
        grouped, dropped = super().merge(models, conflicts=conflicts)
        items: list[StacItem] = []
        for model in models:
            if not isinstance(model, cls):
                continue
            for item in model.stac_items or ():
                if item not in items:
                    items.append(item)
        combined = cls(
            stac_items=tuple(items),
            stac_groupby=grouped.stac_groupby,
        )
        # Items accumulate rather than drop, so differing ones are not a loss.
        dropped.discard("stac_items")
        return combined, dropped
