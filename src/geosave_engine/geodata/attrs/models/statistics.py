"""The summary GDAL stores for one band's pixels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self

from geosave_engine.geodata.attrs.model import AttrsModel

if TYPE_CHECKING:
    import xarray as xr


class BandStatistics(AttrsModel):
    """State the summary GDAL computed over one band's pixels.

    Fields are named for the metadata items themselves, so what the model
    carries is what ``gdalinfo -stats`` wrote. A file carries them only where
    someone computed them, and they describe the pixels as they were then.

    Args:
        STATISTICS_MINIMUM: Smallest value among the band's present pixels.
        STATISTICS_MAXIMUM: Largest value among them.
        STATISTICS_MEAN: Their arithmetic mean.
        STATISTICS_STDDEV: Their population standard deviation.
        STATISTICS_VALID_PERCENT: Share of the band's pixels that are present,
            as a percentage.
        STATISTICS_APPROXIMATE: True where GDAL sampled rather than read every
            pixel.

    Examples:
        >>> ds.gs.attrs.data_vars["B04"].get(BandStatistics).STATISTICS_MEAN
        2001.4357847178
    """

    NAME: ClassVar[str] = "statistics"

    STATISTICS_MINIMUM: float | None = None
    STATISTICS_MAXIMUM: float | None = None
    STATISTICS_MEAN: float | None = None
    STATISTICS_STDDEV: float | None = None
    STATISTICS_VALID_PERCENT: float | None = None
    STATISTICS_APPROXIMATE: bool | None = None

    @classmethod
    def compute(cls, array: xr.DataArray) -> Self:
        """Summarise a variable's pixels, reading every one of them.

        A chunked variable is computed, so this costs a full read. Absent
        pixels have to be NaN already, which is what `Nodata.decode` leaves
        them as; a stored fill still counts as data here.

        Args:
            array: Variable to summarise.

        Returns:
            Model carrying the summary, marked exact.

        Raises:
            ValueError: Every pixel is absent, so there is nothing to
                summarise.

        Examples:
            >>> BandStatistics.compute(ds.B04).STATISTICS_VALID_PERCENT
            99.97
        """
        present = int(array.notnull().sum())
        if not present:
            raise ValueError(
                f"{array.name} holds no present pixel, so it summarises to "
                f"nothing; drop the variable or decode its fill value first"
            )
        return cls(
            STATISTICS_MINIMUM=float(array.min()),
            STATISTICS_MAXIMUM=float(array.max()),
            STATISTICS_MEAN=float(array.mean()),
            STATISTICS_STDDEV=float(array.std()),
            STATISTICS_VALID_PERCENT=100.0 * present / array.size,
            STATISTICS_APPROXIMATE=False,
        )

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge the summaries, keeping only what a join leaves true.

        The joined pixels' extremes are the extremes of the models', so those
        carry over. A mean, a deviation, and a share need each model's pixel
        count, which GDAL never wrote, so they drop instead.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            Model carrying the merged extremes, and the attr keys it dropped.

        Raises:
            TypeError: An object carries a different model.
            ValueError: `models` is empty.
        """
        if not models:
            raise ValueError(f"merging {cls.NAME} needs at least one object")

        summaries: list[Self] = []
        for model in models:
            if model is None:
                continue
            if not isinstance(model, cls):
                raise TypeError(
                    f"merging {cls.NAME} needs {cls.__name__} instances, got "
                    f"{type(model).__name__}"
                )
            summaries.append(model)

        # An object summarising nothing leaves the join's own extremes unknown.
        minimums = [
            s.STATISTICS_MINIMUM for s in summaries if s.STATISTICS_MINIMUM is not None
        ]
        maximums = [
            s.STATISTICS_MAXIMUM for s in summaries if s.STATISTICS_MAXIMUM is not None
        ]
        joined: dict[str, float] = {}
        if len(minimums) == len(models):
            joined["STATISTICS_MINIMUM"] = min(minimums)
        if len(maximums) == len(models):
            joined["STATISTICS_MAXIMUM"] = max(maximums)

        fields: set[str] = set()
        for summary in summaries:
            fields.update(summary.model_fields_set)

        absent = fields - joined.keys()
        values: dict[str, Any] = dict.fromkeys(absent)
        values.update(joined)
        return cls(**values), {cls.attr_keys[name] for name in absent}
