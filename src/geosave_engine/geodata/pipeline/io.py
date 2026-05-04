from datetime import datetime
from pathlib import Path

import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
import xarray as xr


def save_layer(result: dict[str, xr.DataArray], output_dir: str | Path) -> None:
    """Write each DataArray to output_dir/<layer_name>/<lon>_<lat>-<YYYYMMDD>.tif.

    Expects each DataArray to have 'datetime' and 'bbox' attrs set by Pipeline.run().
    """
    output_dir = Path(output_dir)

    for layer_name, da in result.items():
        dt: datetime | None = da.attrs.get("datetime")
        if dt is None:
            raise ValueError(
                f"Layer '{layer_name}' missing 'datetime' attr — "
                "DataArray must come from Pipeline.run()"
            )

        bbox: tuple[float, float, float, float] | None = da.attrs.get("bbox")
        if bbox is None:
            raise ValueError(
                f"Layer '{layer_name}' missing 'bbox' attr — "
                "DataArray must come from Pipeline.run()"
            )

        lon, lat = bbox[0], bbox[1]
        date_str = dt.strftime("%Y%m%d")
        filename = f"{lon:.6f}_{lat:.6f}-{date_str}.tif"

        layer_dir = output_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        da.rio.to_raster(layer_dir / filename)
