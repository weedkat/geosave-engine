from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pystac

from geosave_engine.stac_query import Sentinel2Query
from geosave_engine.stac_query.client import CdseClient
from geosave_engine.utils.datetime import datetime_range_buffer
from geosave_engine.utils.geom import wkt_to_geojson

client = CdseClient()

def mgrs_from_product_id(product_id: str) -> str:
    """Extract MGRS tile code from a Sentinel-2 product ID.

    Example::

        mgrs_from_product_id("S2A_MSIL2A_20190116T210921_N0211_R057_T05QKB_20190116T223213")
        # → "05QKB"
    """
    return product_id.split("_")[-2].lstrip("T")


def save_item(item: pystac.Item, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{item.id}.json").write_text(json.dumps(item.to_dict(), indent=2))


def search_dw_items(row: pd.Series) -> list[pystac.Item]:
    """Return signed Sentinel-2 items matching a DynamicWorld label row.

    L1C items are used instead of L2A because the DynamicWorld labels are based on L1C data.  
    L1C and L2A products have the same spatial and temporal identifiers, so we can reliably 
    match L1C items to the labels via MGRS tile code and sensing datetime.

    Filters by spatial intersection, date range, orbit state, and MGRS tile.
    Items are pre-signed and ready for ``stackstac``.
    """
    dt_range = datetime_range_buffer(row["date"], delta_after=timedelta(days=1))
    geojson = wkt_to_geojson(row["geometry"], row["crs"])
    mgrs = mgrs_from_product_id(row["S2_PRODUCT_ID"])

    query = (
        Sentinel2Query(
            collections=["sentinel-2-l1c"],
            intersects=geojson,
            datetime=dt_range,
        )
        .orbit_state(row["S2_SENSING_ORBIT_DIRECTION"])
    )

    items = []
    
    for item in client.search(query).items():
        if item.properties.get("grid:code") == f"MGRS-{mgrs}":
            items.append(item)
    
    return items



def ingest_from_dw_df(df: pd.DataFrame, output_dir: Path) -> None:
    has_labeler = "labeler" in df.columns

    for _, row in df.iterrows():
        product_id = row["S2_PRODUCT_ID"]
        labeler = str(row["labeler"]) if has_labeler else None
        out_dir = output_dir / labeler if labeler else output_dir
        sensing_dt = product_id.split("_")[2]
        mgrs = mgrs_from_product_id(product_id)
        if out_dir.exists() and any(out_dir.glob(f"*{sensing_dt}*T{mgrs}*.json")):
            print(f"[skip] {product_id}")
            continue

        items = search_dw_items(row)
        if not items:
            print(f"[warn] {product_id} — no match found")
            continue

        for item in items:
            save_item(item, out_dir)
            print(f"[done] {item.id}")


def main() -> None:
    output_dir = Path("data/dw_stac_items")
    xlsx_path = Path(
        "data/dynamicworld_extracted/v1_dw_tile_metadata_for_public_release.xlsx"
    ).resolve()

    df = pd.read_excel(xlsx_path).head(10)  # run a few
    ingest_from_dw_df(df, output_dir / xlsx_path.stem)


if __name__ == "__main__":
    main()
