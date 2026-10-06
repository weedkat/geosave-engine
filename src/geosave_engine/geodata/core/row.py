"""Raster asset operations on one native pandas Series row."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, cast

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

    @property
    def hrefs(self) -> dict[str, str]:
        """Return each data asset's href by key."""
        hrefs = {}
        for key, asset in self._data["assets"].items():
            # Parquet keeps one struct for every row, null where a row has no such asset.
            if asset is not None and (
                asset.get("roles") is None or "data" in asset["roles"]
            ):
                hrefs[key] = asset["href"]
        return hrefs

    def to_raster(self, *, layer: str | None = None, **options: Any) -> xr.Dataset:
        """Read this row's assets as one lazy raster.

        Args:
            layer: Asset key to read. None reads every data asset as one raster.
            **options: Forwarded to `read_raster`. `chunks` defaults to `{}`.

        Returns:
            Dataset cropped to this row's pixel window where it has one.

        Raises:
            KeyError: `layer` names no data asset of this row.
            ValueError: The assets sit on different grids.

        Examples:
            >>> catalog.iloc[0].gs.to_raster().gs.variables
            ('B04', 'B08')
        """
        from geosave_engine.geodata.io import read_raster

        hrefs = self.hrefs
        source = list(hrefs.values()) if layer is None else hrefs[layer]
        return self.crop(read_raster(source, **{"chunks": {}, **options}))

    def to_stack(
        self, *, layers: Collection[str] | str | None = None, **options: Any
    ) -> DataTree:
        """Read this row's assets as a lazy stack, one group per asset.

        Args:
            layers: Asset keys to read. None reads every data asset.
            **options: Forwarded to `read_raster`. `chunks` defaults to `{}`.

        Returns:
            DataTree cropped to this row's pixel window where it has one.

        Raises:
            KeyError: A layer names no data asset of this row.

        Examples:
            >>> samples.iloc[0].gs.to_stack(layers=["optical", "label"]).gs.groups
            ('optical', 'label')
        """
        from geosave_engine.geodata.io import read_stack

        hrefs = self.hrefs
        names = (
            [layers]
            if isinstance(layers, str)
            else list(hrefs if layers is None else layers)
        )
        return self.crop(
            read_stack(
                {name: hrefs[name] for name in names}, **{"chunks": {}, **options}
            )
        )

    def crop[DataT: xr.Dataset | xr.DataArray | xr.DataTree](
        self, data: DataT
    ) -> DataT:
        """Apply this row's pixel window to explicitly supplied native data.

        Args:
            data: Opened raster or stack whose pixels the window addresses.

        Returns:
            Cropped native data; rows without a window return it unchanged.
        """
        from geosave_engine.geodata.transform.chip import crop

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
