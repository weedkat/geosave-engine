"""Packing that decodes stored digital numbers into physical values."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self

from geosave_engine.geodata.attrs.model import AttrsModel

from .stac import shared_asset_fields

if TYPE_CHECKING:
    import pystac


class Packing(AttrsModel):
    """Describe how one variable's stored values decode to physical values.

    Rasters stay in their stored digital numbers while in memory, so a variable
    carrying these fields still holds what its source published. Absence is
    `Nodata`, which packing never touches.

    Args:
        scale_factor: Multiplier applied to each stored value.
        add_offset: Value added after scaling.

    Examples:
        >>> ds.gs.rebase(Packing(scale_factor=1e-4, add_offset=-0.1), target="B04")
        >>> ds.gs.attrs.data_vars["B04"].get(Packing).scale_factor
        0.0001
    """

    NAME: ClassVar[str] = "packing"

    scale_factor: float | None = None
    add_offset: float | None = None

    @classmethod
    def from_stac_asset(cls, items: Sequence[pystac.Item], asset: str) -> Self:
        """Read the packing every item of one load publishes for an asset.

        These fields decide what a stored value means, so a field the items
        disagree on rejects the load: choosing one would silently decode
        the other items' pixels wrong.

        Args:
            items: Items making up one load.
            asset: Asset name to read, which names the variable it loads into.

        Returns:
            Model carrying the packing every item publishes identically, its other
            fields unset.

        Raises:
            ValueError: The items publish a different `scale` or `offset` for the
                asset.

        Examples:
            >>> Packing.from_stac_asset(matched, "B04").scale_factor
            0.0001
        """
        return cls.model_validate(
            shared_asset_fields(
                items,
                asset,
                {"scale": "scale_factor", "offset": "add_offset"},
                on_conflict="reject",
            )
        )

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge packing, refusing disagreements.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            Merged model, and the attr keys it could not keep.

        Raises:
            TypeError: An object carries a different model.
            ValueError: A packing field disagrees or `models` is empty.
        """
        return cls._merge_fields(models, must_agree=cls.model_fields)
