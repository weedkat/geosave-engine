"""Pack a root of per-anchor .zarr stores into one packed SampleStore.

Edit constants below, run directly. Thin wrapper over
geosave_engine.geodata.datastore.ops.zarr_to_litdata.
"""
from __future__ import annotations

from geosave_engine.geodata.datastore.ops import zarr_to_litdata

ROOT = "data/zarr"  # directory to rglob **/*.zarr under, any depth
OUTPUT_DIR = "data/train"  # SampleStore path to write into
LAYER_NAME = "image"  # layer name for an ungrouped (bare-GeoTile) zarr, ignored for a grouped one
REQUIRED_LAYERS = None  # layer names to require; None includes every anchor found
INCLUDE = None  # glob pattern(s) an anchor's relative path must match one of
EXCLUDE = None  # glob pattern(s) an anchor's relative path must not match
MODE = None  # None raises if OUTPUT_DIR already holds a store, "append"/"overwrite" otherwise
ANCHORS_PER_BATCH = 1000  # anchors per write() call, bounds peak memory
CHUNK_SIZE = 1000  # SampleStore's litdata chunk size — exactly one of this/CHUNK_BYTES required


def main() -> None:
    """Scan ROOT, write every matching anchor into OUTPUT_DIR."""
    zarr_to_litdata(
        ROOT,
        OUTPUT_DIR,
        layer_name=LAYER_NAME,
        required_layers=REQUIRED_LAYERS,
        include=INCLUDE,
        exclude=EXCLUDE,
        mode=MODE,
        anchors_per_batch=ANCHORS_PER_BATCH,
        chunk_size=CHUNK_SIZE,
    )


if __name__ == "__main__":
    main()
