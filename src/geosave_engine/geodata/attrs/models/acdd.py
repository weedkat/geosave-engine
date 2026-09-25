"""Discovery metadata describing a whole raster."""

from __future__ import annotations

from typing import ClassVar

from geosave_engine.geodata.attrs.model import AttrsModel


class ACDD(AttrsModel):
    """Who made a raster, what it is, and how it may be used.

    Coverage keys such as `geospatial_bounds` are read off the grid at write
    time and are not fields here.

    Args:
        id: Identifier unique within the dataset's naming authority.
        title: Short human-readable name for the dataset.
        summary: Paragraph describing what the dataset contains.
        keywords: Comma-separated search keywords.
        institution: Organization that produced the dataset.
        creator_name: Person or group responsible for the dataset.
        license: URL or free-text terms for accessing and distributing the dataset.
        source: Method, model, instrument, or observations that produced the data.

    Examples:
        >>> ds.gs.attrs.root.get(ACDD).license
        'CC-BY-4.0'
    """

    NAME: ClassVar[str] = "acdd"

    id: str | None = None
    title: str | None = None
    summary: str | None = None
    keywords: str | None = None
    institution: str | None = None
    creator_name: str | None = None
    license: str | None = None
    source: str | None = None
