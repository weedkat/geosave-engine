"""Exceptions for geodata conditions callers can handle separately."""

__all__ = ["AnchorFetchError", "CollectionNotFoundError"]


class AnchorFetchError(RuntimeError):
    """A STAC search matched no items for an anchor."""


class CollectionNotFoundError(LookupError):
    """A STAC catalog does not publish the requested collection."""
