"""Tags: free-form descriptive strings a caller attached. See Tags for details."""
from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict, model_validator

from geosave_engine.geodata.extensions.base import GeoExtension


class Tags(GeoExtension):
    """Free-form `{key: value}` strings a caller set.

    Every key is its own field — `Tags(source="survey")` and
    `header.rebase(tags={"source": "survey"})` both take a bare mapping,
    not a wrapped `values=` argument, so this namespace holds whatever
    keys the caller gives it directly.

    Raises:
        ValueError: Any value isn't a string.

    Examples:
        >>> Tags(source="survey").model_dump()
        {'source': 'survey'}
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    NAMESPACE: ClassVar[str] = "tags"

    @model_validator(mode="after")
    def _require_string_values(self) -> Tags:
        """Reject a non-string tag value.

        Raises:
            ValueError: A value isn't a string.
        """
        extra = self.__pydantic_extra__ or {}
        bad = {key: value for key, value in extra.items() if not isinstance(value, str)}
        if bad:
            raise ValueError(f"tags must be strings, got non-string values: {bad}")
        return self
