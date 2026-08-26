"""Read and write vector feature files."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

# from_vector dispatches on these; GeoParquet is the only format written back.
VECTOR_SUFFIXES = (".parquet", ".geojson", ".gpkg", ".shp")

# A raster's features ride beside it under this suffix, never inside the store — see sidecar_path.
SIDECAR_SUFFIX = "vector.parquet"


def from_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Read features from a vector file — format from the path's suffix.

    Args:
        path: `.parquet`, `.geojson`, `.gpkg`, or `.shp`, any case.

    Returns:
        Features on the file's own CRS.

    Raises:
        ValueError: `path`'s suffix isn't one of `VECTOR_SUFFIXES`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in VECTOR_SUFFIXES:
        raise ValueError(f"Expected one of {VECTOR_SUFFIXES}, got: {path}")
    return gpd.read_parquet(path) if suffix == ".parquet" else gpd.read_file(path)


def to_geoparquet(gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
    """Write features as GeoParquet, in their own CRS.

    Args:
        gdf: Features to write.
        path: Output `.parquet` path, any case. Parent directories are created.

    Returns:
        The written path.

    Raises:
        ValueError: `path` doesn't end in `.parquet`.
    """
    path = Path(path)
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a .parquet path, got: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    return path


def sidecar_path(path: str | Path, group: str | None = None) -> Path:
    """Where a raster's vector file sits, given the raster's own path.

    Always beside the raster, never inside it — zarr's member scan warns
    about any non-zarr object in the hierarchy, so a sidecar written into a
    store makes it unreadable under warnings-as-errors.

    Args:
        path: The raster file or store path.
        group: Group the pixels were written into, or None for the root.

    Returns:
        `<stem>.vector.parquet`, or `<stem>.<group>.vector.parquet` for a
        grouped write, so rasters sharing one store keep separate files.
    """
    suffix = f"{group}.{SIDECAR_SUFFIX}" if group else SIDECAR_SUFFIX
    return Path(path).with_suffix(f".{suffix}")


def write_sidecar(path: str | Path, gdf: gpd.GeoDataFrame | None, group: str | None = None) -> Path | None:
    """Synchronize a raster's vector sidecar with the pixels just written.

    Args:
        path: The raster path just written.
        gdf: Features to write beside it, or None to leave none.
        group: Group the pixels were written into, or None for the root.

    Returns:
        The written vector path, or None after removing a stale sidecar
        when there are no features.
    """
    sidecar = sidecar_path(path, group)
    if gdf is None:
        sidecar.unlink(missing_ok=True)
        return None
    return to_geoparquet(gdf, sidecar)


def read_sidecar(path: str | Path, group: str | None = None) -> gpd.GeoDataFrame | None:
    """Read the vector sidecar beside a raster, if one was written.

    Args:
        path: The raster file or store path.
        group: Group the pixels were read from, or None for the root.

    Returns:
        Features on their own CRS, or None when no sidecar is there.
    """
    sidecar = sidecar_path(path, group)
    return from_vector(sidecar) if sidecar.exists() else None
