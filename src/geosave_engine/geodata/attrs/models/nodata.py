"""The stored value a variable's absent pixels hold."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import Field, model_validator

from geosave_engine.geodata.attrs.model import AttrsModel, values_agree

from .stac import shared_asset_fields

if TYPE_CHECKING:
    import pystac


class Nodata(AttrsModel):
    """Name the stored value marking one variable's absent pixels.

    Args:
        fill_value: Stored value marking absence, written as `_FillValue`.
        nodata: odc's spelling of the same value, mirroring `fill_value` so the
            two can never name different pixels. Either one marks absence;
            setting both differently is refused.

    Raises:
        ValueError: `fill_value` and `nodata` are set to different values.

    Examples:
        >>> ds.gs.rebase(Nodata(fill_value=0), target="B04")
        >>> ds.gs.attrs.data_vars["B04"].get(Nodata).fill_value
        0
    """

    NAME: ClassVar[str] = "nodata"

    fill_value: int | float | None = Field(default=None, alias="_FillValue")
    nodata: int | float | None = None

    @model_validator(mode="after")
    def _mirror_both_spellings(self) -> Self:
        """Mirror the fill value across both spellings, refusing a disagreement.

        odc reads `nodata` ahead of `_FillValue`, so a stale one would decide
        which pixels are absent. Either spelling marks absence and the other
        follows it.

        Returns:
            The model with both spellings naming one value.

        Raises:
            ValueError: The two spellings are set to different values.
        """
        if (
            self.fill_value is not None
            and self.nodata is not None
            and not values_agree(self.fill_value, self.nodata)
        ):
            raise ValueError(
                f"_FillValue {self.fill_value!r} and nodata {self.nodata!r} name "
                f"different absent pixels; set one of them"
            )

        # Assigning only on a difference keeps validate_assignment from recursing.
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
        """Read the absent value every item of one load publishes for an asset.

        Args:
            items: Items making up one load.
            asset: Asset name to read, which names the variable it loads into.

        Returns:
            Model carrying the value every item publishes identically, its other
            fields unset.

        Raises:
            ValueError: The items publish a different `nodata` for the asset.

        Examples:
            >>> Nodata.from_stac_asset(matched, "B04").fill_value
            0
        """
        return cls.model_validate(
            shared_asset_fields(
                items, asset, {"nodata": "fill_value"}, on_conflict="reject"
            )
        )

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge the fill value, refusing disagreements.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            Merged model, and the attr keys it could not keep.

        Raises:
            TypeError: An object carries a different model.
            ValueError: The fill value disagrees or `models` is empty.
        """
        return cls._merge_fields(models, must_agree=cls.model_fields)
