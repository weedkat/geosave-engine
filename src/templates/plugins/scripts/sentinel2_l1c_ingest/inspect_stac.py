"""Inspect available bands for a STAC collection via CDSE.

Usage:
    python inspect_stac.py --collection sentinel-2-l1c --bbox 10 45 11 46 --date 2023-06-01/2023-06-30
    python inspect_stac.py --collection sentinel-2-l1c --bbox 10 45 11 46 --date 2023-06-01/2023-06-30 --out bands.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from geosave_engine.geodata.stac_client import CdseClient
from geosave_engine.geodata.core.query import BaseStacQuery


def main() -> None:
    parser = argparse.ArgumentParser(description="List available STAC bands for a collection.")
    parser.add_argument("--collection", required=True, help="STAC collection ID (e.g. sentinel-2-l1c)")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), required=True)
    parser.add_argument("--date", required=True, help="Date range: YYYY-MM-DD/YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    client = CdseClient()
    query  = BaseStacQuery(
        collections=[args.collection],
        bbox=tuple(args.bbox),
        datetime=args.date,
    )

    items = client.search(query)
    if not items:
        print(f"No items found for collection={args.collection!r} bbox={args.bbox} date={args.date!r}")
        return

    item   = items[0]
    bands  = sorted(item.assets.keys())

    result = {
        "collection": args.collection,
        "item_id":    item.id,
        "datetime":   item.properties.get("datetime"),
        "bands":      bands,
    }

    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote to {args.out}")


if __name__ == "__main__":
    main()
