"""Rebuild one .zarr per sample from a packed SampleStore — reverses zarr_to_litdata.

Edit constants below, run directly. Thin wrapper over
geosave_engine.geodata.datastore.ops.litdata_to_zarr.
"""
from __future__ import annotations

from geosave_engine.geodata.datastore.ops import litdata_to_zarr
from geosave_engine.geodata.datastore.sample import SampleStore

STORE_PATH = "data/train"  # SampleStore folder to read from
CHUNK_SIZE = 1000  # must match how STORE_PATH was written — SampleStore needs it even for a read-only open
OUTPUT_DIR = "data/zarr_rebuilt"  # root directory to write .zarr stores under
OVERWRITE = True  # False raises instead of replacing an existing anchor


def main() -> None:
    """Read STORE_PATH, write one .zarr per sample under OUTPUT_DIR."""
    store = SampleStore(STORE_PATH, chunk_size=CHUNK_SIZE)
    litdata_to_zarr(store, OUTPUT_DIR, overwrite=OVERWRITE)


if __name__ == "__main__":
    main()
