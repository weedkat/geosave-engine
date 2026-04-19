"""Sentinel-2 L1C preprocessing pipeline aligned with DynamicWorld labels.

Pipeline
--------
1. Build an index of saved STAC item JSONs keyed by (sensing_datetime, mgrs_tile)
2. Pair each xlsx row to its item via that index
3. Per sample: stackstac → clip to label geometry → normalize → save GeoTIFF

Bands (in model input order, all resampled to 10m)
---------------------------------------------------
B02  Blue       (10m native)
B03  Green      (10m native)
B04  Red        (10m native)
B05  Red Edge 1 (20m → 10m bilinear)
B06  Red Edge 2 (20m → 10m bilinear)
B07  Red Edge 3 (20m → 10m bilinear)
B08  NIR        (10m native)
B11  SWIR 1     (20m → 10m bilinear)
B12  SWIR 2     (20m → 10m bilinear)

Reference: Brown et al. (2022) Dynamic World, Scientific Data
           https://www.nature.com/articles/s41597-022-01307-4
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from dask.diagnostics import ProgressBar
from tqdm import tqdm

import numpy as np
import pandas as pd
import pystac
import rioxarray  # noqa: F401 — registers .rio accessor on xarray
import shapely.wkt
import stackstac
import xarray as xr
from rasterio.enums import Resampling


# 30th / 70th percentiles of log-transformed L1C reflectances.
# Source: google/dynamicworld single_image_runner.ipynb
# https://github.com/google/dynamicworld/blob/master/single_image_runner.ipynb
NORM_PERCENTILES = np.array([
    [1.7417268007636313, 2.023298706048351],
    [1.7261204997060209, 2.038905204308012],
    [1.6798346251414997, 2.179592821212937],
    [1.7734969472909623, 2.2890068333026603],
    [2.289154079164943,  2.6171674549378166],
    [2.382939712192371,  2.773418590375327],
    [2.3828939530384052, 2.7578332604178284],
    [2.1952484264967844, 2.789092484314204],
    [1.554812948247501,  2.4140534947492487],
])

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_item(path: Path) -> pystac.Item:
    return pystac.Item.from_dict(json.loads(path.read_text()))


def build_item_index(items_dir: Path) -> dict[tuple[str, str], pystac.Item]:
    """Index saved items by (sensing_datetime, mgrs_tile).

    Key is derived from the item ID, e.g.:
        S2A_MSIL1C_20190116T210921_N0500_R057_T05QKB_... → ("20190116T210921", "05QKB")
    """
    index: dict[tuple[str, str], pystac.Item] = {}
    for path in items_dir.rglob("*.json"):
        item = load_item(path)
        parts = item.id.split("_")
        sensing_dt = parts[2]
        mgrs = parts[-2].lstrip("T")
        index[(sensing_dt, mgrs)] = item
    return index


def pair_rows_with_items(
    df: pd.DataFrame,
    item_index: dict[tuple[str, str], pystac.Item],
) -> Iterator[tuple[pd.Series, pystac.Item]]:
    """Yield (row, item) for each xlsx row that has a matching saved item."""
    for _, row in df.iterrows():
        product_id = row["S2_PRODUCT_ID"]
        sensing_dt = product_id.split("_")[2]
        mgrs = product_id.split("_")[-2].lstrip("T")
        item = item_index.get((sensing_dt, mgrs))
        if item is None:
            print(f"[warn] no item for {product_id}")
            continue
        yield row, item


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

def stack_item(item: pystac.Item, bands: list[str], epsg: int, resolution: int = 10) -> xr.DataArray:
    """Lazy-load one item into a (band, y, x) DataArray via stackstac."""
    da = stackstac.stack(
        [item],
        assets=bands,
        epsg=epsg,
        resolution=resolution,
        resampling=Resampling.bilinear,
        chunksize=512, # set this to
    )
    return da.squeeze("time")


def normalize(da: xr.DataArray) -> xr.DataArray:
    """Apply DynamicWorld L1C normalization → output in (0, 1)."""
    x = np.log(da * 0.005 + 1)
    p30 = NORM_PERCENTILES[:, 0].reshape(-1, 1, 1) # broadcast (band, y, x)
    p70 = NORM_PERCENTILES[:, 1].reshape(-1, 1, 1) # broadcast (band, y, x)
    x = (x - p30) / p70
    x = np.exp(x * 5 - 1)
    return x / (x + 1)


def clip_to_geometry(da: xr.DataArray, geometry_wkt: str, epsg: int) -> xr.DataArray:
    """Clip to the DW label polygon.

    Both the DataArray and the WKT geometry are in the same UTM CRS so no
    reprojection is needed. rioxarray accepts shapely geometries directly.
    """
    geom = shapely.wkt.loads(geometry_wkt)
    return da.rio.write_crs(epsg).rio.clip([geom])


def save_as_geotiff(da: xr.DataArray, path: Path) -> None:
    """Compute the lazy DataArray and write it to a GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with ProgressBar():
        da.rio.to_raster(path, compress="lzw", lock=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_sample(
    row: pd.Series,
    item: pystac.Item,
    bands: list[str],
    output_dir: Path,
) -> None:
    epsg = int(str(row["crs"]).split(":")[-1])
    out_path = output_dir / f"{item.id}.tif"
    if out_path.exists():
        print(f"[skip] {item.id}")
        return

    t0 = time.perf_counter()
    da = stack_item(item, bands=bands, epsg=epsg)
    da = clip_to_geometry(da, row["geometry"], epsg)
    da = normalize(da)
    t1 = time.perf_counter()
    save_as_geotiff(da, out_path)
    t2 = time.perf_counter()
    print(f"[done] {item.id}  build={t1-t0:.1f}s  save={t2-t1:.1f}s")


def main() -> None:

    ITEMS_DIR = Path("data/dw_stac_items")
    OUTPUT_DIR = Path("data/preprocessed")

    xlsx_path = Path(
        "data/dynamicworld_extracted/v1_dw_tile_metadata_for_public_release.xlsx"
    ).resolve()

    df = pd.read_excel(xlsx_path)
    item_index = build_item_index(ITEMS_DIR)
    print(f"Loaded {len(item_index)} unique items")

    pairs = list(pair_rows_with_items(df, item_index))
    for row, item in tqdm(pairs, desc="preprocessing", unit="sample"):
        labeler = str(row["labeler"]) if "labeler" in row.index else None
        out_dir = OUTPUT_DIR / labeler if labeler else OUTPUT_DIR
        process_sample(row, item, bands=BANDS, output_dir=out_dir)


if __name__ == "__main__":
    main()
