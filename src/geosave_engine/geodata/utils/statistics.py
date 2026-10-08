"""Eager pixel summaries shared by array and raster accessors."""

from collections.abc import Hashable, Mapping

from dask.base import compute
import pandas as pd
import xarray as xr

from geosave_engine.geodata.attrs import Nodata


def statistics(variables: Mapping[Hashable, xr.DataArray]) -> pd.DataFrame:
    """Summarise variables while computing shared pixel sources only once.

    Args:
        variables: Row names mapped to variables. Each variable is reduced
            across all axes, excluding NaN and its own fill value.

    Returns:
        DataFrame in input order with minimum, maximum, mean, population
        stddev, and valid_percent columns. An empty input keeps these columns.

    Raises:
        ValueError: A variable holds no present pixels.
    """
    reductions = []
    for band in variables.values():
        fill = Nodata.from_attrs(band.attrs)
        values = (
            band
            if fill is None or fill.fill_value is None
            else band.where(band != fill.fill_value)
        )
        present = values.notnull().sum()
        reductions.append(
            xr.Dataset(
                {
                    "present": present,
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "mean": values.mean(),
                    "stddev": values.std(),
                    "valid_percent": 100.0 * present / values.size,
                }
            )
        )

    columns = ["minimum", "maximum", "mean", "stddev", "valid_percent"]
    rows = {}
    for name, summary in zip(variables, compute(*reductions), strict=True):
        if not int(summary["present"]):
            raise ValueError(
                f"{name} holds no present pixel, so it summarises to "
                f"nothing; drop the band or give it pixels that are not fill"
            )
        row = {}
        for column in columns:
            row[column] = float(summary[column])
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=pd.Index(columns))
