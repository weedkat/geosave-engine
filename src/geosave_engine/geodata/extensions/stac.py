"""StacItems: which STAC items a raster came from. See StacItems for details."""
from __future__ import annotations

from datetime import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar, Self, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from geosave_engine.geodata.extensions.base import GeoExtension
from geosave_engine.geodata.utils.datetime import naive_utc

if TYPE_CHECKING:
    from geosave_engine.geodata.extensions.timespec import TimeSpec


class StacItemRecord(BaseModel):
    """One STAC item's identity, common metadata and chosen extension properties.

    Common metadata is a closed spec, so it is typed. Extension properties
    are open-ended and land in `properties`, keyed exactly as STAC publishes
    them; which ones are read is a `StacSourceConfig` decision.

    Args:
        id: Item ID.
        datetime: Item's own timestamp, stored naive UTC. Falls back to
            `start_datetime`/`end_datetime` for a collection dating items by
            validity range rather than instant.
        collection: Owning collection ID.
        title: Human-readable item title.
        description: Human-readable item description.
        start_datetime: Start of the item's validity range.
        end_datetime: End of the item's validity range.
        created: When the item was first produced.
        updated: When the item was last revised.
        platform: Platform name, e.g. `"sentinel-2b"`.
        instruments: Instrument names.
        constellation: Constellation name.
        mission: Mission name.
        gsd: Ground sample distance in meters.
        license: SPDX license identifier or `"proprietary"`.
        providers: Organizations that produced or host the item.
        keywords: Free-form keywords the provider tagged the item with.
        roles: Roles the provider declared for the item.
        properties: Extension properties read off the item, keyed as STAC
            publishes them, e.g. `{"eo:cloud_cover": 4.2}`.

    Examples:
        >>> record.properties["eo:cloud_cover"]
        4.2
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    datetime: dt
    collection: str | None = None
    title: str | None = None
    description: str | None = None
    start_datetime: dt | None = None
    end_datetime: dt | None = None
    created: dt | None = None
    updated: dt | None = None
    platform: str | None = None
    instruments: tuple[str, ...] | None = None
    constellation: str | None = None
    mission: str | None = None
    gsd: float | None = None
    license: str | None = None
    providers: tuple[dict[str, Any], ...] | None = None
    keywords: tuple[str, ...] | None = None
    roles: tuple[str, ...] | None = None
    properties: dict[str, Any] = {}

    @field_validator("datetime", "start_datetime", "end_datetime", "created", "updated")
    @classmethod
    def _store_naive(cls, value: dt | None) -> dt | None:
        """Normalize a timestamp to naive UTC.

        Args:
            value: Timestamp, aware or naive.

        Returns:
            Same instant, naive UTC. None passes through.
        """
        return None if value is None else naive_utc(value)


class StacItems(GeoExtension):
    """Which STAC items a raster was built from.

    Records are flat and self-dating, never keyed by the array's own time
    labels, so resampling and windowing cannot desynchronize them. Which
    items back a step is derived from that step's own bucket bounds.

    Args:
        items: Every source item, in any order.

    Examples:
        >>> raster.stac.at(raster.times[0], raster.timespec)
        (StacItemRecord(id='S2A_...'), StacItemRecord(id='S2B_...'))
    """

    NAMESPACE: ClassVar[str] = "stac"

    items: tuple[StacItemRecord, ...] = ()

    def between(self, start: dt, end: dt) -> Self:
        """Narrow to the items acquired inside a span, both ends inclusive.

        Args:
            start: Span start.
            end: Span end.

        Returns:
            New StacItems over the items inside the span.
        """
        start, end = naive_utc(start), naive_utc(end)
        return type(self)(items=tuple(item for item in self.items if start <= item.datetime <= end))

    def at(self, time: dt, spec: TimeSpec | None = None) -> tuple[StacItemRecord, ...]:
        """Items backing one time step.

        Args:
            time: One `time` coord label.
            spec: The raster's own `timespec`, naming the bucket that label
                stands for. None matches the label exactly.

        Returns:
            Items whose own acquisition time falls in that label's bucket.
        """
        if spec is None:
            stamp = naive_utc(time)
            return tuple(item for item in self.items if item.datetime == stamp)
        start, end = spec.bounds(np.array([time], dtype="datetime64[ns]"))[0]
        return self.between(start, end).items

    @classmethod
    def combine(cls, values: Sequence[GeoExtension | None]) -> Self | None:
        """Union every array's items, deduplicated by item ID.

        Args:
            values: Extensions being composed, in composition order.

        Returns:
            StacItems over every distinct item, in first-seen order. None
            when any input lacks provenance, since a partial union would
            claim complete provenance for the result.
        """
        if any(value is None for value in values):
            return None
        merged: dict[str, StacItemRecord] = {}
        for value in values:
            if not isinstance(value, cls):
                raise TypeError(f"{cls.__name__}.combine expected only {cls.__name__} values")
            for item in value.items:
                merged.setdefault(item.id, item)
        return cls(items=tuple(merged.values()))
