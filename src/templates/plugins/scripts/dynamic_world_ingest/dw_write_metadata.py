"""Write per-dataset metadata layers into the workspace manifest.

Workflow
--------
Run once after ``dw_ingest_cdse.py`` has finished ingesting all splits.

Layers written (idempotent — skips if already present):
  - ``metadata_dynamicworld``: DW class table (class_id, class_name, source_class_id, ignore, RGB).
  - ``metadata_cloud_mask``:   binary mask class table (clear / masked).
  - ``bands_sentinel_2_l1c``:  band-index → name / wavelength / resolution.
"""
from __future__ import annotations

from geosave_engine.geodata.datasets import Sentinel2L1C
from geosave_engine.utils.geodata.manifest import write_attribute_layer

from dw_ingest_cdse import BANDS, DW_CLASSES, WORKSPACE_DATA

CLOUD_MASK_CLASSES: list[dict] = [
    {"class_id": 0, "class_name": "clear",  "source_class_id": 0, "ignore": False, "color_r": 0,   "color_g": 0,   "color_b": 0},
    {"class_id": 1, "class_name": "masked", "source_class_id": 1, "ignore": True,  "color_r": 255, "color_g": 255, "color_b": 255},
]


def _s2l1c_band_rows(bands: list[str]) -> list[dict]:
    return [
        {
            "band_index":    i + 1,
            "band_name":     band,
            "wavelength_um": Sentinel2L1C.wavelengths[band],
            "resolution":    Sentinel2L1C.resolutions[band],
        }
        for i, band in enumerate(bands)
    ]


def main() -> None:
    manifest_path = WORKSPACE_DATA / "manifest.gpkg"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path}; run dw_ingest_cdse.py first"
        )

    write_attribute_layer(DW_CLASSES,            manifest_path, layer="metadata_dynamicworld")
    write_attribute_layer(CLOUD_MASK_CLASSES,    manifest_path, layer="metadata_cloud_mask")
    write_attribute_layer(_s2l1c_band_rows(BANDS), manifest_path, layer="bands_sentinel_2_l1c")


if __name__ == "__main__":
    main()
