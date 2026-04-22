"""Ingest DynamicWorld labels via CDSE using SentinelL1CService."""
from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import dask
import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from base_ingest import query_from_dw_row
from geosave_engine.ingestion import (
    SentinelL1CService,
    append_to_manifest,
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_s2c_mask,
    load_or_init_manifest,
)
from geosave_engine.stac_query import CdseClient, Sentinel2Query, StacClient
from geosave_engine.utils.geom import wkt_to_geojson


BANDS: list[str] = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
S2CLOUDLESS_BANDS: list[str] = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
NAN_THRESHOLD = 0.01


def _log_nan_diagnostics(key: str, item_id: str, da: xr.DataArray, *, threshold: float) -> None:
    total_nan = float(np.isnan(da.values).mean())
    per_band = {
        str(b): float(np.isnan(da.sel(band=b).values).mean())
        for b in da.band.values
    }
    ordered = sorted(per_band.items(), key=lambda kv: kv[1], reverse=True)
    top = ", ".join(f"{band}={ratio:.1%}" for band, ratio in ordered[:5])
    tqdm.write(
        f"  [nan-diag] {key} item={item_id} total_nan={total_nan:.2%} "
        f"(threshold={threshold:.2%}) top_bands: {top}"
    )


def _save_mask_layer(arr: np.ndarray, path: Path, da: xr.DataArray, svc: SentinelL1CService) -> None:
    layer = xr.DataArray(arr.astype(np.uint8), dims=["y", "x"], coords={"y": da.y, "x": da.x})
    layer = layer.rio.write_crs(da.rio.crs)
    layer = layer.rio.write_transform(da.rio.transform())
    svc.save(layer, path)


def process_sample(
    row: pd.Series,
    item,
    query: Sentinel2Query,
    svc: SentinelL1CService,
    input_dir: Path,
    mask_dir: Path,
    tci_dir: Path,
) -> dict | None:
    key    = str(row["dw_id"])
    sun_az = float(item.properties.get("view:sun_azimuth", 0.0))

    utm_bounds = shapely.from_wkt(str(row["geometry"])).bounds

    for attempt in range(1, 4):
        try:
            lazy = svc.load_item(item, query, resolution=10, utm_bounds=utm_bounds)
            with ProgressBar():
                all_da = dask.compute(lazy)[0]
            da = all_da.sel(band=BANDS)
            if np.isnan(da.values).mean() > NAN_THRESHOLD:
                _log_nan_diagnostics(key, item.id, da, threshold=NAN_THRESHOLD)
                raise RuntimeError("excessive NaN -- possible corrupt read")
            break
        except Exception as exc:
            msg = str(exc)
            if "Read failed" in msg or "opj_get_decoded_tile" in msg:
                tqdm.write(f"[skip] {key} -- JP2 decode failure: {exc}")
                return None
            if attempt == 3:
                tqdm.write(f"[skip] {key} -- fetch failed after 3 attempts: {exc}")
                return None
            tqdm.write(f"  [retry {attempt}] {exc}")
            time.sleep(2 ** attempt)

    s2c    = compute_s2c_mask(all_da.sel(band=S2CLOUDLESS_BANDS).values)
    cdi    = compute_cdi_mask(
        all_da.sel(band="B07").values,
        all_da.sel(band="B08").values,
        all_da.sel(band="B8A").values,
    )
    b10    = compute_b10_mask(all_da.sel(band="B10").values)
    cloud  = s2c & cdi & b10
    shadow = build_shadow_mask(cloud, sun_az, resolution=10)
    mask   = cloud | shadow

    input_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    tci_dir.mkdir(parents=True, exist_ok=True)

    svc.save(da, input_dir / f"{key}.tif")
    tqdm.write(f"[save] input/{key}.tif")

    _save_mask_layer(mask, mask_dir / f"{key}.tif", da, svc)
    tqdm.write(f"[save] mask/{key}.tif  (masked={mask.mean():.1%})")
    tqdm.write(f"  s2c={s2c.mean():.1%}  cdi={cdi.mean():.1%}  b10={b10.mean():.1%}  shadow={shadow.mean():.1%}")

    rgb = (da.sel(band=["B04", "B03", "B02"]).clip(0, 0.3) / 0.3 * 255).astype(np.uint8)
    svc.save(rgb, tci_dir / f"{key}.tif")
    tqdm.write(f"[save] tci/{key}.tif")

    labeler  = str(row["labeler"]) if "labeler" in row.index else None

    # GeoJSON stores geometry in WGS84
    geometry = shapely.geometry.shape(wkt_to_geojson(str(row["geometry"]), str(row["crs"])))

    return {
        "dw_id":            key,
        "labeler":          labeler,
        "item_id":          item.id,
        "sensing_datetime": item.properties.get("datetime"),
        "sun_azimuth":      sun_az,
        "masked_pct":       float(mask.mean()),
        "s2c_pct":          float(s2c.mean()),
        "cdi_pct":          float(cdi.mean()),
        "b10_pct":          float(b10.mean()),
        "shadow_pct":       float(shadow.mean()),
        "input_path":       str(input_dir / f"{key}.tif"),
        "mask_path":        str(mask_dir  / f"{key}.tif"),
        "tci_path":         str(tci_dir   / f"{key}.tif"),
        "split":            None,
        "geometry":         geometry,
    }


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    output_dir    = Path("data/processed")
    cache_dir     = Path("data/dw_stac_items_cdse/v1_dw_tile_metadata_for_public_release")
    xlsx_path     = Path("data/dynamicworld_extracted/v1_dw_tile_metadata_for_public_release.xlsx").resolve()
    manifest_path = output_dir / "manifest.geojson"

    svc = SentinelL1CService(cast(StacClient, CdseClient()), cache_dir=cache_dir)
    gdf = load_or_init_manifest(manifest_path)
    df  = pd.read_excel(xlsx_path).iloc[:20]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ingesting (CDSE)", unit="row"):
        key     = str(row["dw_id"])
        labeler = str(row["labeler"]) if "labeler" in row.index else None

        if key in gdf["dw_id"].values:
            tqdm.write(f"[skip] {key} -- already in manifest")
            continue

        sub       = labeler or ""
        input_dir = output_dir / "input" / sub
        mask_dir  = output_dir / "mask"  / sub
        tci_dir   = output_dir / "tci"   / sub

        query = query_from_dw_row(row, include_orbit_state=True)
        items = svc.search_items(query)
        if not items:
            tqdm.write(f"[warn] {key} -- no match")
            continue

        for item in items:
            try:
                record = process_sample(row, item, query, svc, input_dir, mask_dir, tci_dir)
            except Exception as exc:
                tqdm.write(f"[fail] {key} - {item.id} -- {exc}")
                record = None
            if record is not None:
                gdf = append_to_manifest(gdf, record, manifest_path)
                break


if __name__ == "__main__":
    main()
