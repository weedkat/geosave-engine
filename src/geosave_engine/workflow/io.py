"""Open workflow raster inputs and persist completed local stacks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils import io


@contextmanager
def open_rasters(
    raw: xr.DataTree | Mapping[str, xr.Dataset | str | Path] | str | Path,
) -> Iterator[xr.DataTree]:
    """Open named rasters or a saved stack as one native DataTree.

    Caller-owned Datasets and DataTrees remain open. Datasets and stacks opened
    from paths are closed when the context exits, including after a partial
    mapping fails to open.

    Args:
        raw: Existing stack, named raster objects or paths, or a saved stack path.

    Yields:
        Flat DataTree containing one named raster per group.

    Raises:
        TypeError: An input is neither a supported native object nor a path.
        ValueError: A path suffix names no supported raster or stack format.
    """
    # Close files opened here; leave caller-owned objects open.
    with ExitStack() as opened:
        if isinstance(raw, xr.DataTree):
            yield raw
            return

        if isinstance(raw, Mapping):
            rasters: dict[str, xr.Dataset] = {}
            for name, source in raw.items():
                if isinstance(source, xr.Dataset):
                    rasters[name] = source
                elif isinstance(source, str | Path):
                    raster = io.read_raster(source, chunks="auto")
                    opened.callback(raster.close)
                    rasters[name] = raster
                else:
                    raise TypeError(
                        f"Raster {name!r} must be an xarray Dataset or path, got "
                        f"{type(source).__name__}"
                    )
            yield stack(rasters)
            return

        if isinstance(raw, str | Path):
            tree = io.read_stack(raw, chunks="auto")
            opened.callback(tree.close)
            yield tree
            return

        raise TypeError(
            "raw must be an xarray DataTree, a mapping of named rasters, or a "
            f"saved stack path, got {type(raw).__name__}"
        )


def write_stack(tree: xr.DataTree, path: str | Path) -> str:
    """Write a raster stack completely to a new local Zarr store.

    Args:
        tree: Native raster-stack DataTree to persist.
        path: New local path ending in ``.zarr``.

    Returns:
        String form of the completed destination path.

    Raises:
        FileExistsError: The destination already exists.
        TypeError: `tree` is not a DataTree.
        ValueError: The destination is remote or does not end in ``.zarr``.
    """
    destination = _destination(path)

    # Publish the destination only after every raster has been written.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        io.zarr.write(tree, staged, compute=True, overwrite=False)
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        staged.rename(destination)
    return str(destination)


def _destination(path: str | Path) -> Path:
    """Return a new local Zarr path, rejecting remote or existing destinations."""
    if "://" in str(path):
        raise ValueError("Raster stack output requires a local path")
    destination = Path(path)
    if destination.suffix != ".zarr":
        raise ValueError("Raster stack output must end in .zarr")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    return destination
