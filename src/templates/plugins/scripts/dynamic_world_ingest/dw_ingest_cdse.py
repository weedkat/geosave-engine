"""Ingest DynamicWorld-labelled Sentinel-2 L1C patches via CDSE."""
from __future__ import annotations

import os

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_TCP_KEEPALIVE", "YES")
os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_HTTP_RETRY_COUNT", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from geosave_engine.geodata.ingestion import Sentinel2L1CIngestor
from geosave_engine.geodata.processing.masking import (
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_s2c_mask,
)
from geosave_engine.geodata.stac_client.cdse_client import CdseClient
from geosave_engine.utils.geodata.manifest import (
    append_to_manifest,
    load_manifest,
    write_class_meta,
)


BANDS:             list[str] = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
S2CLOUDLESS_BANDS: list[str] = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
_ALL_BANDS:        list[str] = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
NODATA_SKIP_THRESHOLD = 0.50

DW_CLASSES = [
    {"class_id": 0,   "class_name": "water",              "source_class_id": 1, "ignore": False, "color_r": 65,  "color_g": 155, "color_b": 223},
    {"class_id": 1,   "class_name": "trees",              "source_class_id": 2, "ignore": False, "color_r": 57,  "color_g": 125, "color_b": 73},
    {"class_id": 2,   "class_name": "grass",              "source_class_id": 3, "ignore": False, "color_r": 136, "color_g": 176, "color_b": 83},
    {"class_id": 3,   "class_name": "flooded_vegetation", "source_class_id": 4, "ignore": False, "color_r": 122, "color_g": 135, "color_b": 198},
    {"class_id": 4,   "class_name": "crops",              "source_class_id": 5, "ignore": False, "color_r": 228, "color_g": 150, "color_b": 53},
    {"class_id": 5,   "class_name": "shrub_scrub",        "source_class_id": 6, "ignore": False, "color_r": 223, "color_g": 195, "color_b": 90},
    {"class_id": 6,   "class_name": "built_area",         "source_class_id": 7, "ignore": False, "color_r": 196, "color_g": 40,  "color_b": 27},
    {"class_id": 7,   "class_name": "bare_ground",        "source_class_id": 8, "ignore": False, "color_r": 165, "color_g": 155, "color_b": 143},
    {"class_id": 8,   "class_name": "snow_ice",           "source_class_id": 9, "ignore": False, "color_r": 179, "color_g": 159, "color_b": 225},
    {"class_id": 255, "class_name": "masked",             "source_class_id": 0, "ignore": True,  "color_r": 0,   "color_g": 0,   "color_b": 0},
]


def _build_tiff_index(root: Path) -> dict[str, Path]:
    """Map dw_id → TIFF path, preferring label_* files when both variants exist."""
    index: dict[str, Path] = {}
    for p in root.rglob("*.tif"):
        stem = p.stem
        key  = stem[6:] if stem.startswith("label_") else stem
        if key not in index or p.stem.startswith("label_"):
            index[key] = p
    return index


def _build_mask(
    da_all: xr.DataArray,
    sun_azimuth: float,
) -> tuple[np.ndarray, dict[str, float]]:
    s2c    = compute_s2c_mask(da_all.sel(band=S2CLOUDLESS_BANDS).values)
    cdi    = compute_cdi_mask(
        da_all.sel(band="B07").values,
        da_all.sel(band="B08").values,
        da_all.sel(band="B8A").values,
    )
    b10    = compute_b10_mask(da_all.sel(band="B10").values)
    cloud  = s2c & cdi & b10
    shadow = build_shadow_mask(cloud, sun_azimuth, resolution=10)
    return cloud | shadow, {
        "s2c":    float(s2c.mean()),
        "cdi":    float(cdi.mean()),
        "b10":    float(b10.mean()),
        "shadow": float(shadow.mean()),
    }


def _save_outputs(
    svc: Sentinel2L1CIngestor,
    key: str,
    da: xr.DataArray,
    final_mask: np.ndarray,
    input_dir: Path,
    mask_dir: Path,
    tci_dir: Path,
) -> tuple[str, str, str]:
    input_path = input_dir / f"{key}.tif"
    mask_path  = mask_dir  / f"{key}.tif"
    tci_path   = tci_dir   / f"{key}.tif"

    svc.save(da, input_path)

    mask_layer = xr.DataArray(
        final_mask.astype(np.uint8), dims=["y", "x"], coords={"y": da.y, "x": da.x}
    )
    mask_layer = mask_layer.rio.write_crs(da.rio.crs)
    mask_layer = mask_layer.rio.write_transform(da.rio.transform())
    svc.save(mask_layer, mask_path)

    tci = (da.sel(band=["B04", "B03", "B02"]).clip(0, 0.3) / 0.3 * 255).astype(np.uint8)
    svc.save(tci, tci_path)
    return str(input_path), str(mask_path), str(tci_path)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    output_dir = Path("data/processed")
    input_dir  = output_dir / "input"
    mask_dir   = output_dir / "mask"
    tci_dir    = output_dir / "tci"
    for d in (input_dir, mask_dir, tci_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.gpkg"
    xlsx_path     = Path("data/dynamicworld_extracted/v1_dw_tile_metadata_for_public_release.xlsx").resolve()
    dw_root       = Path("data/dynamicworld_extracted")

    svc        = Sentinel2L1CIngestor(CdseClient())
    df         = pd.read_excel(xlsx_path).iloc[100:120]
    tiff_index = _build_tiff_index(dw_root)

    write_class_meta(DW_CLASSES, manifest_path)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="ingesting (CDSE)", unit="row"):
        key = str(row.dw_id)

        if manifest_path.exists():
            gdf = load_manifest(manifest_path)
            if key in gdf["dw_id"].values:
                tqdm.write(f"[skip] {key} -- already in manifest")
                continue

        tiff_path = tiff_index.get(key)
        if tiff_path is None:
            tqdm.write(f"[skip] {key} -- no TIFF found")
            continue

        try:
            result = svc.from_tiff(tiff_path, bands=_ALL_BANDS)
        except Exception as exc:
            tqdm.write(f"[warn] {key} -- error: {exc}")
            continue

        if result is None:
            tqdm.write(f"[skip] {key} -- no data found")
            continue

        all_da, tiff_meta = result

        nodata_mask = np.isnan(all_da.sel(band=BANDS).values).all(axis=0)
        nodata_pct  = float(nodata_mask.mean())
        if nodata_pct > NODATA_SKIP_THRESHOLD:
            tqdm.write(f"[skip] {key} -- too much nodata ({nodata_pct:.2%})")
            continue

        sun_azimuth      = float(all_da.attrs.get("sun_azimuth", 0.0))
        item_ids         = [str(x) for x in all_da.attrs.get("item_ids", [])]
        sensing_datetime = all_da.attrs.get("sensing_datetime")

        da_all_filled     = all_da.fillna(0.0)
        da                = da_all_filled.sel(band=BANDS)
        cloud_mask, stats = _build_mask(da_all_filled, sun_azimuth)
        final_mask        = cloud_mask | nodata_mask

        input_path, mask_path, tci_path = _save_outputs(
            svc, key, da, final_mask, input_dir, mask_dir, tci_dir,
        )

        record = {
            "dw_id":              key,
            "split":              "train",
            "labeler":            row.labeler,
            "crs":                tiff_meta.crs,
            "hemisphere":         row.hemisphere,
            "biome":              row.biome,
            "biome_name":         row.biome_name,
            "sensing_datetime":   sensing_datetime,
            "Bare_Ground":        row.Bare_Ground,
            "Built_Area":         row.Built_Area,
            "Clouds":             row.Clouds,
            "Crops":              row.Crops,
            "Flooded_Vegetation": row.Flooded_Vegetation,
            "Grass":              row.Grass,
            "Scrub":              row.Scrub,
            "Snow_Ice":           row.Snow_Ice,
            "Trees":              row.Trees,
            "Water":              row.Water,
            "item_id":            "|".join(item_ids),
            "sun_azimuth":        sun_azimuth,
            "masked_pct":         float(final_mask.mean()),
            "s2c_pct":            stats["s2c"],
            "cdi_pct":            stats["cdi"],
            "b10_pct":            stats["b10"],
            "shadow_pct":         stats["shadow"],
            "label_path":         str(tiff_path),
            "input_path":         input_path,
            "mask_path":          mask_path,
            "tci_path":           tci_path,
            "geometry":           tiff_meta.geometry,
        }

        append_to_manifest(record, manifest_path, native_crs=tiff_meta.crs)


if __name__ == "__main__":
    main()
