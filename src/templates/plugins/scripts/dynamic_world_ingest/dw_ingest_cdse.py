"""Ingest DynamicWorld-labelled Sentinel-2 L1C patches via CDSE.

Workflow
--------


Output written to ``DATA``:

    data/
    ├── cloud_mask/{train,val,test}/*.tif    # uint8 single-band union mask
    ├── sentinel_2_l1c/{train,val,test}/*.tif
    └── manifest.gpkg

DynamicWorld labels are NOT copied — they are renamed in-place to the
canonical ``dw_<lon>_<lat>-<YYYYMMDD>.tif`` filename so that TorchGeo's
``RasterLabel`` regex can discover them recursively.  Point
``RasterLabel(paths=...)`` at the extracted split directory; nested folder
structure is preserved.
"""
from __future__ import annotations

import os

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN",       "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX",                "YES")
os.environ.setdefault("GDAL_HTTP_TCP_KEEPALIVE",            "YES")
os.environ.setdefault("GDAL_HTTP_UNSAFESSL",                "YES")
os.environ.setdefault("GDAL_HTTP_TIMEOUT",                  "120")
os.environ.setdefault("GDAL_HTTP_RETRY_COUNT",              "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY",              "2")
os.environ.setdefault("GDAL_NUM_THREADS",                   "1")

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from geosave_engine.geodata.core.errors import IngestionError, IngestionErrorReason
from geosave_engine.geodata.ingestion import Sentinel2L1CIngestor
from geosave_engine.geodata.processing.masking import (
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_s2c_mask,
)
from geosave_engine.geodata.stac_client.cdse import CdseClient
from geosave_engine.utils.cli.fs_ops import ensure_split_dirs
from geosave_engine.utils.geodata.manifest import (
    append_geo_layer,
    list_manifest_layers,
    load_geo_layer,
)
from geosave_engine.utils.geodata.masks import union_masks, write_single_band_mask
from geosave_engine.utils.geodata.tiff import build_tiff_filename, read_tiff_metadata

DW_CLASSES: list[dict] = [
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

# Output band order written into sentinel_2_l1c GeoTIFFs.
BANDS:             list[str] = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
# Bands required by s2cloudless.
S2CLOUDLESS_BANDS: list[str] = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
# Superset fetched from STAC — covers BANDS + S2CLOUDLESS_BANDS + CDI inputs.
ALL_BANDS:         list[str] = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

SPLITS             = ("train", "val", "test")
OUTPUT_DATASETS    = ("sentinel_2_l1c", "cloud_mask")   # dynamicworld stays in-place
NODATA_SKIP_THRESHOLD = 0.50
TIFF_PREFIX        = "dw"

WORKSPACE_DATA = Path("data")


def _already_ingested(manifest_path: Path, fname: str) -> bool:
    if "sentinel_2_l1c" not in list_manifest_layers(manifest_path):
        return False
    gdf = load_geo_layer(manifest_path, "sentinel_2_l1c")
    return "filename" in gdf.columns and fname in gdf["filename"].astype(str).values


def _build_cloud_components(da_filled, sun_azimuth: float):
    s2c    = compute_s2c_mask(da_filled.sel(band=S2CLOUDLESS_BANDS).values).astype(bool)
    cdi    = compute_cdi_mask(
        da_filled.sel(band="B07").values,
        da_filled.sel(band="B08").values,
        da_filled.sel(band="B8A").values,
    ).astype(bool)
    b10    = compute_b10_mask(da_filled.sel(band="B10").values).astype(bool)
    shadow = build_shadow_mask(s2c & cdi & b10, sun_azimuth, resolution=10).astype(bool)
    return s2c, cdi, b10, shadow


def _append_records(
    manifest_path: Path,
    *,
    fname: str,
    split: str,
    tiff_meta,
    da_attrs: dict,
    cloud_union: np.ndarray,
    s2c: np.ndarray,
    cdi: np.ndarray,
    b10: np.ndarray,
    shadow: np.ndarray,
    nodata_pct: float,
) -> None:
    common = {"filename": fname, "split": split, "geometry": tiff_meta.geometry}
    append_geo_layer(
        {
            **common,
            "sensing_datetime": da_attrs.get("sensing_datetime"),
            "item_ids":         "|".join(str(x) for x in da_attrs.get("item_ids", [])),
            "sun_azimuth":      float(da_attrs.get("sun_azimuth", 0.0)),
            "crs":              tiff_meta.crs,
        },
        manifest_path,
        layer="sentinel_2_l1c",
        native_crs=tiff_meta.crs,
    )
    append_geo_layer(
        {
            **common,
            "masked_pct": float(cloud_union.mean()),
            "s2c_pct":    float(s2c.mean()),
            "cdi_pct":    float(cdi.mean()),
            "b10_pct":    float(b10.mean()),
            "shadow_pct": float(shadow.mean()),
            "nodata_pct": nodata_pct,
        },
        manifest_path,
        layer="cloud_mask",
        native_crs=tiff_meta.crs,
    )


def ingest_split(
    label_paths: Iterable[Path],
    split: str,
    *,
    output_root: Path = WORKSPACE_DATA,
    manifest_path: Path | None = None,
    svc: Sentinel2L1CIngestor | None = None,
) -> None:
    """Ingest a list of DW label TIFFs into ``output_root`` under ``split``.

    ``label_paths`` can be any iterable — glob, CSV reader, XLSX rows, etc.
    The caller decides which paths belong to this split.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS!r}; got {split!r}")

    out_paths     = ensure_split_dirs(output_root, OUTPUT_DATASETS, SPLITS)
    manifest_path = manifest_path or output_root / "manifest.gpkg"
    svc           = svc or Sentinel2L1CIngestor(CdseClient())

    label_list = list(label_paths)
    for label_src in tqdm(label_list, total=len(label_list), desc=f"ingesting [{split}]", unit="tif"):
        label_src = Path(label_src)

        tiff_meta = read_tiff_metadata(label_src)
        fname     = build_tiff_filename(tiff_meta, TIFF_PREFIX)
        if _already_ingested(manifest_path, fname):
            continue
        if label_src.name != fname:
            label_src = label_src.rename(label_src.parent / fname)

        try:
            all_da, tiff_meta = svc.from_tiff(label_src, bands=ALL_BANDS)
        except IngestionError as exc:
            level = "skip" if exc.reason is IngestionErrorReason.NO_ITEMS_FOUND else "warn"
            tqdm.write(f"[{level}] {label_src.name} -- {exc.reason.value}: {exc}")
            continue

        nodata_mask = np.isnan(all_da.sel(band=BANDS).values).all(axis=0)
        nodata_pct  = float(nodata_mask.mean())
        if nodata_pct > NODATA_SKIP_THRESHOLD:
            tqdm.write(f"[skip] {label_src.name} -- too much nodata ({nodata_pct:.2%})")
            continue

        da_filled = all_da.fillna(0.0)
        da        = da_filled.sel(band=BANDS)

        s2c, cdi, b10, shadow = _build_cloud_components(da_filled, float(all_da.attrs.get("sun_azimuth", 0.0)))
        cloud_union = union_masks(s2c, cdi, b10, shadow, nodata_mask)

        svc.save(da, out_paths[("sentinel_2_l1c", split)] / fname)
        write_single_band_mask(cloud_union, da, out_paths[("cloud_mask", split)] / fname, svc)

        _append_records(
            manifest_path,
            fname=fname,
            split=split,
            tiff_meta=tiff_meta,
            da_attrs=all_da.attrs,
            cloud_union=cloud_union,
            s2c=s2c,
            cdi=cdi,
            b10=b10,
            shadow=shadow,
            nodata_pct=nodata_pct,
        )


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    # Drive from the extracted DW directory layout — swap for CSV/XLSX/glob as needed.
    dw_extracted = Path("data/dynamicworld")
    for split in SPLITS:
        split_root = dw_extracted / split
        if not split_root.exists():
            continue
        ingest_split(sorted(split_root.rglob("*.tif")), split)


if __name__ == "__main__":
    main()
