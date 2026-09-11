"""Packing that decodes stored digital numbers into physical values."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import Field, model_validator

from geosave_engine.geodata.attrs.model import AttrsModel, values_agree

from .stac import shared_asset_fields

if TYPE_CHECKING:
    import pystac


class Packing(AttrsModel):
    """Describe how one variable's stored values decode to physical values.

    Rasters stay in their stored digital numbers while in memory. Readers
    opting into `mask_and_scale` apply these fields.

    Args:
        scale_factor: Multiplier applied to each stored value.
        add_offset: Value added after scaling.
        fill_value: Stored value marking missing pixels, written as
            `_FillValue`.
        nodata: odc's spelling of the same value, mirroring `fill_value` so
            the two can never name different pixels. Either one declares the
            fill; stating both differently is refused.

    Raises:
        ValueError: `fill_value` and `nodata` state different values.

    Examples:
        >>> ds.gs.rebase(
        ...     Packing(scale_factor=1e-4, add_offset=-0.1, fill_value=0),
        ...     target="B04",
        ... )
        >>> ds.gs.attrs.get(Packing, target="B04").fill_value
        0
    """

    NAME: ClassVar[str] = "packing"

    scale_factor: float | None = None
    add_offset: float | None = None
    fill_value: int | float | None = Field(default=None, alias="_FillValue")
    nodata: int | float | None = None

    @model_validator(mode="after")
    def _sync_nodata(self) -> Self:
        """Mirror the fill value across both spellings, refusing a disagreement.

        odc reads `nodata` ahead of `_FillValue`, so a stale one would decide
        which pixels are absent. Either spelling declares the fill and the
        other follows it.

        Returns:
            The model with both spellings naming one value.

        Raises:
            ValueError: The two spellings state different values.
        """
        if (
            self.fill_value is not None
            and self.nodata is not None
            and not values_agree(self.fill_value, self.nodata)
        ):
            raise ValueError(
                f"_FillValue {self.fill_value!r} and nodata {self.nodata!r} name "
                f"different absent pixels; state one of them"
            )

        # Assigning only on a difference keeps validate_assignment from recursing,
        # and a NaN fill disagrees with itself under plain `!=`.
        mirrored = self.fill_value if self.fill_value is not None else self.nodata
        if not values_agree(self.fill_value, mirrored):
            self.fill_value = mirrored
        if not values_agree(self.nodata, mirrored):
            self.nodata = mirrored
        elif "fill_value" in self.model_fields_set:
            # A cleared fill must clear odc's spelling too, so it has to be written.
            self.model_fields_set.add("nodata")
        return self

    @classmethod
    def from_stac_asset(cls, items: Sequence[pystac.Item], asset: str) -> Self:
        """Read the packing every item of one load states for an asset.

        These fields decide what a stored value means, so a field the items
        state differently rejects the load: choosing one would silently decode
        the other items' pixels wrong.

        Args:
            items: Items making up one load.
            asset: Asset name to read, which names the variable it loads into.

        Returns:
            Model carrying the packing every item states identically, its other
            fields unset.

        Raises:
            ValueError: The items state a different `scale`, `offset`, or
                `nodata` for the asset.

        Examples:
            >>> Packing.from_stac_asset(matched, "B04").scale_factor
            0.0001
        """
        return cls.model_validate(
            shared_asset_fields(
                items,
                asset,
                {
                    "scale": "scale_factor",
                    "offset": "add_offset",
                    "nodata": "fill_value",
                },
                on_conflict="reject",
            )
        )

    @classmethod
    def combine(cls, sides: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Combine packing, refusing disagreements.

        Args:
            sides: This model as each side of the join stated it, in call
                order, at least one, None where a side did not state it.

        Returns:
            Combined model, and the attr keys it could not keep.

        Raises:
            TypeError: A side holds a different model.
            ValueError: A packing field disagrees or `models` is empty.
        """
        return cls._combine_fields(sides, must_agree=cls.model_fields)
