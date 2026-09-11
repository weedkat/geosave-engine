"""Render hints naming the Dataset variables used as display channels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

from geosave_engine.geodata.attrs.model import AttrsModel


class RenderHints(AttrsModel):
    """Declare Dataset variables used as RGB display channels.

    Names variables rather than describing them, so the hint is only true of a
    Dataset still carrying all three.

    Args:
        rgb_variables: Three Dataset variable names ordered red, green, blue.

    Examples:
        >>> ds.gs.rebase(
        ...     RenderHints(rgb_variables=("B04", "B03", "B02"))
        ... )
    """

    NAME: ClassVar[str] = "render"

    rgb_variables: tuple[str, str, str] | None = None

    @classmethod
    def combine(cls, sides: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Combine display hints, dropping disagreements.

        Args:
            sides: This model as each side of the join stated it, in call
                order, at least one, None where a side did not state it.

        Returns:
            Model carrying the fields every side states alike, and the
            attr keys it could not keep.

        Raises:
            TypeError: A side holds a different model.
            ValueError: `models` is empty.
        """
        return cls._combine_fields(sides, must_agree=())
