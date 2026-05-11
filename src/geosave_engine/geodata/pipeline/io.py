from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Annotated

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
import xarray as xr

_PRIMITIVE_TYPES = (str, int, float, bool, type(None))
_RESERVED: frozenset[str] = frozenset({"layers", "processed"})


# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------

LayerName = Annotated[str, "Name of the layer"]
Roles = Literal["label", "image", "mask"]


@dataclass
class LayerSchema:
    name: LayerName
    role: Roles
    dtype: str
    num_bands: int | None = None
    num_class: int | None = None


@dataclass
class BandMeta:
    """Metadata for one band of an image layer."""
    name: str
    wavelength: float | None = None
    resolution: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassMeta:
    """Metadata for one class of a label layer. Use class_name='ignore_index' for void."""
    id: int
    name: str
    color: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskMeta:
    """Metadata for one region of a mask layer."""
    id: int
    name: str
    color: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serialize_attr(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, _PRIMITIVE_TYPES):
        return value
    return json.dumps(value, default=str)


def _make_filename(da: xr.DataArray) -> str:
    bbox: tuple[float, float, float, float] = da.attrs["bbox"]
    dt: datetime = da.attrs["datetime"]
    lon, lat = bbox[0], bbox[1]
    return f"{lon:.6f}_{lat:.6f}-{dt.strftime('%Y%m%d')}.tif"


def _load_sheets(path: Path, sheet_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_names, engine="openpyxl")  # type: ignore[return-value]


def _write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def _flatten_meta(items: list[BandMeta] | list[ClassMeta] | list[MaskMeta]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        row = asdict(item)
        row.update(row.pop("extra", {}))
        rows.append(row)
    return rows


def _manifest_load(path: Path) -> ManifestSheet:
    if not path.exists():
        return ManifestSheet()

    sheets = _load_sheets(path)
    if sheets.get("layers") is None:
        raise ValueError("Manifest missing required 'layers' sheet")
    if sheets.get("processed") is None:
        raise ValueError("Manifest missing required 'processed' sheet")

    return ManifestSheet.from_dict(sheets)


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------


@dataclass
class ManifestSheet:
    layers: list[LayerSchema] = field(default_factory=list)
    metadata: dict[LayerName, list[dict[str, Any]]] = field(default_factory=dict)
    records: dict[LayerName, list[dict[str, Any]]] = field(default_factory=dict)
    processed: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, pd.DataFrame]:
        sheets: dict[str, pd.DataFrame] = {
            "layers": pd.DataFrame([asdict(layer) for layer in self.layers]),
            "processed": pd.Series(list(self.processed), name="filename").to_frame(),
        }
        sheets.update({f"{k}_meta": pd.DataFrame(v) for k, v in self.metadata.items()})
        sheets.update({k: pd.DataFrame(v) for k, v in self.records.items()})
        return sheets

    @classmethod
    def from_dict(cls, sheets: dict[str, pd.DataFrame]) -> ManifestSheet:
        layers = [LayerSchema(**row) for row in sheets.pop("layers").to_dict(orient="records")]
        processed = set(sheets.pop("processed")["filename"].tolist())

        metadata, records = {}, {}
        for name, df in sheets.items():
            data = df.to_dict(orient="records")
            if name.endswith("_meta"):
                metadata[name.removesuffix("_meta")] = data
            else:
                records[name] = data

        return ManifestSheet(layers=layers, records=records, metadata=metadata, processed=processed)


# ---------------------------------------------------------------------------
# Public writer
# ---------------------------------------------------------------------------


class ManifestWriter:
    """Stateful xlsx manifest writer with write-through, crash recovery, and skip detection.

    Sheet layout:
        layers              — one row per declared layer (name, role, dtype, num_bands, num_class)
        {layer}_meta        — band or class definitions (BandMeta / ClassMeta / MaskMeta rows)
        {layer}             — one row per tile (filename, split, da.attrs)
        processed           — one row per fully written tile (filename)
    """

    def __init__(self, output_dir: str | Path, manifest_name: str = "manifest.xlsx") -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / manifest_name
        self.sheets = _manifest_load(self.path)

    def flush(self) -> None:
        """Write current manifest state to disk."""
        tmp_path = self.path.with_suffix(".tmp")
        _write_sheets(tmp_path, self.sheets.to_dict())
        tmp_path.replace(self.path)

    def create_layer(
        self,
        name: LayerName,
        role: Roles,
        meta: list[BandMeta] | list[ClassMeta] | list[MaskMeta],
        dtype: str | None = None,
    ) -> None:
        """Declare a new layer and write its metadata sheet. Upserts by name."""
        if name in _RESERVED:
            raise ValueError(f"Layer name {name!r} is reserved")

        dtype = dtype or ("float32" if role == "image" else "uint8")
        num_bands = len(meta) if role == "image" else None
        num_class = len(meta) if role in ("label", "mask") else None

        schema = LayerSchema(name=name, role=role, dtype=dtype, num_bands=num_bands, num_class=num_class)
        self.sheets.layers = [lyr for lyr in self.sheets.layers if lyr.name != name]
        self.sheets.layers.append(schema)
        self.sheets.metadata[name] = _flatten_meta(meta)

        self.flush()

    def record_tile(self, result: dict[str, xr.DataArray], split: str | None = None, source_id: str | None = None) -> None:
        """Record tile metadata for each declared layer. Upserts by filename. No file IO.

        Args:
            source_id: Original source file stem. When provided, added to processed sheet
                so is_written(source_id) returns True.
        """
        declared = {lyr.name for lyr in self.sheets.layers}
        result_keys = set(result.keys())

        missing = declared - result_keys
        if missing:
            raise ValueError(f"Result missing declared layers: {missing}")

        extra = result_keys - declared
        if extra:
            warnings.warn(f"Result contains undeclared layers (skipped): {extra}")

        if not declared:
            return

        filename = _make_filename(result[next(iter(declared))])

        for name in declared:
            da = result[name]
            record: dict[str, Any] = {"filename": filename, "split": split}
            for k, v in da.attrs.items():
                record[k] = _serialize_attr(v)
            existing = [r for r in self.sheets.records.get(name, []) if r.get("filename") != filename]
            existing.append(record)
            self.sheets.records[name] = existing

        if source_id is not None:
            self.sheets.processed.add(source_id)
        self.flush()

    def write_tile(self, result: dict[str, xr.DataArray], split: str | None = None, source_id: str | None = None) -> None:
        """Write tile rasters and record manifest. Upserts by filename.

        Args:
            source_id: Original source file stem (e.g. tiff_path.stem). When provided,
                added to the processed sheet so is_written(source_id) returns True.
        """
        declared = {lyr.name for lyr in self.sheets.layers}
        result_keys = set(result.keys())

        missing = declared - result_keys
        if missing:
            raise ValueError(f"Result missing declared layers: {missing}")

        extra = result_keys - declared
        if extra:
            warnings.warn(f"Result contains undeclared layers (skipped): {extra}")

        layer_map = {lyr.name: lyr for lyr in self.sheets.layers}
        for name in declared:
            layer_dir = self.output_dir / name / split if split is not None else self.output_dir / name
            da = result[name].astype(layer_map[name].dtype)
            save_tile(da, layer_dir)

        self.record_tile({name: result[name] for name in declared}, split=split, source_id=source_id)

    def remove_layer(self, name: LayerName) -> None:
        """Remove layer, its metadata, tile records, and scrub processed."""
        self.sheets.layers = [lyr for lyr in self.sheets.layers if lyr.name != name]
        self.sheets.metadata.pop(name, None)
        self.sheets.records.pop(name, None)

        if self.sheets.records:
            filename_sets = [{r["filename"] for r in records} for records in self.sheets.records.values()]
            self.sheets.processed = set.intersection(*filename_sets)
        else:
            self.sheets.processed = set()

        self.flush()

    def is_written(self, filename: str) -> bool:
        """Return True only if filename appears in the processed sheet (all layers written)."""
        return filename in self.sheets.processed


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def save_tile(da: xr.DataArray, output_dir: str | Path) -> Path:
    """Write a single DataArray to output_dir/<filename>.tif. Returns the written path.

    Args:
        da: DataArray with 'datetime' and 'bbox' attrs set by Pipeline.run().
        output_dir: Directory to write the raster into.

    Returns:
        Path to the written file.
    """
    dt: datetime | None = da.attrs.get("datetime")
    if dt is None:
        raise ValueError(
            "DataArray missing 'datetime' attr — must come from Pipeline.run()"
        )
    bbox: tuple[float, float, float, float] | None = da.attrs.get("bbox")
    if bbox is None:
        raise ValueError(
            "DataArray missing 'bbox' attr — must come from Pipeline.run()"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _make_filename(da)
    da.rio.to_raster(path)
    return path


def compute_class_pcts(da: xr.DataArray) -> dict[str, float]:
    """Compute per-class pixel percentages from a label DataArray.

    Args:
        da: Integer label array.

    Returns:
        Dict mapping 'class_{value}' to fraction of total pixels.
    """
    values = da.values.flatten()
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    total = values.size
    return {f"class_{int(v)}": int(c) / total for v, c in zip(unique, counts)}
