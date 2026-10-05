"""Raster asset operations on one native pandas Series row."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, cast

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from geosave_engine.geodata import DataTree


@pd.api.extensions.register_series_accessor("gs")
class GeoRow:
    """Open a selected row's assets or apply its explicit pixel window.

    Args:
        data: Native Series containing a catalog or reference row.
    """

    def __init__(self, data: pd.Series):
        self._data = data

    def to_xarray(self, *, layers: Collection[str] | str | None = None) -> DataTree:
        """Open selected raster assets and apply any stored pixel window lazily.

        Args:
            layers: Asset names to open; None opens every available asset.

        Returns:
            Native DataTree containing the selected raster groups.

        Raises:
            KeyError: The row names no assets or a selected asset is absent.

        Examples:
            >>> sample = catalog.iloc[0].gs.to_xarray()
        """
        from geosave_engine.geodata.io.assets import read

        if "assets" not in self._data:
            raise KeyError("this row carries no 'assets' naming rasters")
        assets = self._data["assets"]
        if not isinstance(assets, Mapping):
            raise KeyError("the row names no assets")
        opened = read(assets, layers=layers)
        try:
            result = self.crop(opened)
        except BaseException:
            opened.close()
            raise
        if result is not opened:
            result.set_close(opened.close)
        return result

    def crop[DataT: xr.Dataset | xr.DataArray | xr.DataTree](
        self, data: DataT
    ) -> DataT:
        """Apply this row's pixel window to explicitly supplied native data.

        Args:
            data: Opened raster or stack whose pixels the window addresses.

        Returns:
            Cropped native data; rows without a window return it unchanged.
        """
        from geosave_engine.geodata.transform.window import crop

        row = self._data.to_dict()
        if "row_off" not in row or bool(pd.isna(row["row_off"])):
            return data
        mode = row.get("padding_mode", "constant")
        value = row.get("padding_value", float("nan"))
        fill = value if mode == "constant" and not bool(pd.isna(value)) else None
        return cast(
            "DataT",
            crop(
                data,
                (int(row["row_off"]), int(row["col_off"])),
                (int(row["height"]), int(row["width"])),
                padding=row.get("padding", ((0, 0), (0, 0))),
                mode=mode,
                constant_value=fill,
            ),
        )
