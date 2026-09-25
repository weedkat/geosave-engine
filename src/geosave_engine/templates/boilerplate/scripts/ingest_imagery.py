"""Ingest imagery for every label tile, mirroring the label tree's own structure.

A label's filename date plus its own grid becomes the StacSource anchor;
output COGs land at the same relative path under out-root. A manifest there
tracks status, so a rerun after a network failure skips what already succeeded.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import typer
from dotenv import load_dotenv

import geosave_engine as gs
from geosave_engine.geodata.stac import StacClient, StacSource

_MANIFEST_COLUMNS = ("label", "status", "output", "error", "ingested_at")

Catalog = Literal["cdse", "planetary_computer", "element84"]
_CATALOGS = {
    "cdse": StacClient.cdse,
    "planetary_computer": StacClient.planetary_computer,
    "element84": StacClient.element84,
}


class IngestManifest:
    """Resumable per-label CSV/Excel ledger, one row per label at all times.

    Every `record` call writes the file immediately: a label's row must
    survive a crash right after it's recorded, since resuming depends on it.

    Args:
        path: Manifest file path, ending in `.csv` or `.xlsx`.
        frame: Rows already on disk, or an empty frame for a new run.
    """

    def __init__(self, path: Path, frame: pd.DataFrame) -> None:
        """Bind a loaded frame to its path; construct through `open`."""
        self._path = path
        self._frame = frame

    @classmethod
    def open(cls, path: Path) -> IngestManifest:
        """Load the manifest at `path`, or start an empty one.

        Args:
            path: Manifest file path, ending in `.csv` or `.xlsx`.

        Returns:
            Manifest holding every row previously written to `path`.

        Raises:
            ValueError: `path`'s suffix is neither `.csv` nor `.xlsx`.
        """
        if path.suffix not in (".csv", ".xlsx"):
            raise ValueError(f"manifest path must end in .csv or .xlsx, got {path}")
        if not path.exists():
            return cls(path, pd.DataFrame(columns=list(_MANIFEST_COLUMNS)))
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        return cls(path, frame)

    def is_recorded(self, label: str) -> bool:
        """Report whether a prior run already attempted `label`.

        Args:
            label: Label's path relative to raw-root, posix-separated.

        Returns:
            True where any row names `label`, whether it succeeded or failed.
            A resume retries neither, since a failure names a label this
            ingest could not handle; delete its row to attempt it again.
        """
        return bool((self._frame["label"] == label).any())

    def record(
        self,
        label: str,
        *,
        status: Literal["ok", "failed"],
        output: str = "",
        error: str = "",
    ) -> None:
        """Write one row for `label`, replacing any earlier row for it.

        Args:
            label: Label's path relative to raw-root, posix-separated.
            status: `"ok"` or `"failed"`.
            output: Written COG path, relative to out-root. Empty on failure.
            error: Failure detail. Empty on success.
        """
        row = {
            "label": label,
            "status": status,
            "output": output,
            "error": error,
            "ingested_at": datetime.now(UTC).isoformat(),
        }
        self._frame = pd.concat(
            [self._frame[self._frame["label"] != label], pd.DataFrame([row])],
            ignore_index=True,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.suffix == ".csv":
            self._frame.to_csv(self._path, index=False)
        else:
            self._frame.to_excel(self._path, index=False)


def find_labels(raw_root: Path) -> list[Path]:
    """Every .tif under raw_root, sorted.

    Args:
        raw_root: Label tree root.

    Returns:
        Label paths in the tree.
    """
    return sorted(raw_root.rglob("*.tif"))


def label_anchor(label_path: Path) -> gs.GeoAnchor:
    """Read a label's grid and the period it covers.

    Args:
        label_path: Label .tif carrying its own datetime tag.

    Returns:
        The label's anchor over its grid and that period.

    Raises:
        ValueError: The label carries no datetime tag.
    """
    anchor = gs.read_raster(label_path).gs.anchor
    if anchor.timespan is None:
        raise ValueError(f"{label_path.name} carries no datetime tag")
    return anchor


def ingest_label(
    label_path: Path,
    label: str,
    out_root: Path,
    source: StacSource,
    manifest: IngestManifest,
    *,
    red: str,
    green: str,
    blue: str,
) -> None:
    """Load imagery for one label's anchor and save it as a COG at the mirrored path.

    One GeoTIFF holds one instant, so a load matching several keeps its first
    scene and warns for the rest. That scene's `time` floors to whole seconds,
    since a capture's sub-second time cannot fit a GeoTIFF datetime tag.

    Args:
        label_path: Label .tif supplying the anchor.
        label: `label_path`'s path relative to raw-root, posix-separated —
            the manifest key and the relative output path.
        out_root: Imagery tree root.
        source: Bound StacSource to load from.
        manifest: Ledger recording this label's outcome.
        red: Loaded band drawn as the red channel.
        green: Loaded band drawn as the green channel.
        blue: Loaded band drawn as the blue channel.

    Warns:
        UserWarning: The load matched more than one scene.
    """
    out_path = out_root / label
    try:
        anchor = label_anchor(label_path)
        loaded = source.load(anchor)
        if loaded.sizes["time"] > 1:
            warnings.warn(
                f"{label} matched {loaded.sizes['time']} scenes; writing the "
                f"first on the time axis and dropping the rest",
                stacklevel=2,
            )
        scene = loaded.isel(time=0)
        scene = scene.assign_coords(time=scene.time.dt.floor("s"))
        scene = scene.gs.write_rgb(red=red, green=green, blue=blue)
        scene.gs.to_cog(out_path, overwrite=True)
    except Exception as error:
        manifest.record(label, status="failed", error=str(error))
    else:
        manifest.record(
            label, status="ok", output=out_path.relative_to(out_root).as_posix()
        )


def main(
    raw_root: Path = Path("data/labels"),
    out_root: Path = Path("data/imagery"),
    collection: str = "sentinel-2-l2a",
    bands: list[str] = ["B04", "B03", "B02"],  # noqa: B006 — one-shot CLI process
    red: str = "B04",
    green: str = "B03",
    blue: str = "B02",
    catalog: Catalog = "planetary_computer",
    manifest_name: str = "manifest.csv",
) -> None:
    """Ingest imagery for every label under raw_root, resuming from out_root's manifest.

    Args:
        raw_root: Label tree root, walked for `.tif` files.
        out_root: Imagery tree root; COGs mirror the label tree under it.
        collection: STAC collection to load imagery from.
        bands: Band names `collection` publishes, catalog-specific — Planetary
            Computer's sentinel-2-l2a publishes e.g. `"B04"`, where CDSE's
            publishes `"B04_10m"`.
        red: Band in `bands` drawn as the red channel, so a GIS composes a
            true colour.
        green: Band in `bands` drawn as the green channel.
        blue: Band in `bands` drawn as the blue channel.
        catalog: STAC catalog to search.
        manifest_name: Manifest filename under out_root — `.csv` or `.xlsx`.

    Raises:
        BadParameter: `red`, `green`, or `blue` names a band absent from
            `bands`, or two of them name the same band.

    Examples:
        Typer names each parameter after itself, underscores as dashes, and
        takes a repeated `--bands` for the list::

            $ python scripts/ingest_imagery.py \\
                --raw-root examples/data/dw_label \\
                --out-root examples/data/dw_imagery

            $ python scripts/ingest_imagery.py --help

        A catalog spelling its bands its own way needs them named, and reads
        its credentials from a `.env` beside the working directory::

            $ python scripts/ingest_imagery.py --catalog cdse \\
                --bands B04_10m --bands B03_10m --bands B02_10m \\
                --red B04_10m --green B03_10m --blue B02_10m
    """
    channels = dict(zip(("red", "green", "blue"), (red, green, blue), strict=True))
    unknown = sorted(set(channels.values()) - set(bands))
    if unknown:
        raise typer.BadParameter(f"{unknown} are not in --bands {bands}")
    if len(set(channels.values())) < len(channels):
        raise typer.BadParameter(
            f"one band cannot draw two channels, but these name {channels}; "
            f"give each channel its own band"
        )

    load_dotenv()
    gs.configure_gdal(gdal_disable_readdir_on_open=True, gdal_http_max_retry=3)

    client = _CATALOGS[catalog]()
    source = client.source(collection).set_config(bands=bands)
    manifest = IngestManifest.open(out_root / manifest_name)

    for label_path in find_labels(raw_root):
        label = label_path.relative_to(raw_root).as_posix()
        if manifest.is_recorded(label):
            continue
        ingest_label(
            label_path,
            label,
            out_root,
            source,
            manifest,
            red=red,
            green=green,
            blue=blue,
        )


if __name__ == "__main__":
    typer.run(main)
