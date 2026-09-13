"""Discovery metadata describing a whole raster."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self

from geosave_engine.geodata.attrs.model import AttrsModel

if TYPE_CHECKING:
    import pystac


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
        >>> ds.gs.attrs.get(ACDD).license
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

    @classmethod
    def from_collection(cls, collection: pystac.Collection) -> Self:
        """Read discovery metadata off a STAC collection.

        The collection's first provider becomes `institution` and its keywords
        join into one comma-separated string. `creator_name` and `source` stay
        unset, which a collection publishes no counterpart for.

        Args:
            collection: Collection the pixels were loaded from.

        Returns:
            Model carrying the collection's own metadata.

        Examples:
            >>> ACDD.from_collection(client.collection("sentinel-2-l2a")).institution
            'ESA'
        """
        providers = collection.providers or []
        return cls(
            id=collection.id,
            title=collection.title,
            summary=collection.description,
            keywords=", ".join(collection.keywords) if collection.keywords else None,
            institution=providers[0].name if providers else None,
            license=collection.license,
        )

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge discovery metadata, dropping disagreements.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            Model carrying the fields every object agreed on, and the
            attr keys it could not keep.

        Raises:
            TypeError: An object carries a different model.
            ValueError: `models` is empty.
        """
        return cls._merge_fields(models, must_agree=())
