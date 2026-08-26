"""LitDataStore: many samples packed into one litdata store.

Thin wrapper around litdata's optimize()/StreamingDataset — domain-blind,
takes a plain dict sample or any custom fn. GeoStack auto-detection is
blocked pending GeoStack's own redesign around GeoRaster.
"""
from __future__ import annotations

import functools
import inspect
import json
import pickle
import tempfile
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, Literal, Sequence, TypedDict, Unpack, Mapping, cast

import numpy as np

import pandas as pd
from litdata import optimize
from litdata.processing.readers import BaseReader
from litdata.streaming import StreamingDataset
from litdata.streaming.fs_provider import FsProvider, _get_fs_provider
from litdata.streaming.item_loader import BaseItemLoader
from litdata.utilities.encryption import Encryption

from geosave_engine.geodata.extensions import ArraySpec
from geosave_engine.geodata.spatial import (
    DEFAULT_LAYER,
    GeoTile,
    GeoTileStack,
    decode_anchor,
    encode_anchor,
    numpy_context,
    read_windows,
)
from geosave_engine.geodata.utils.datastore import (
    CONTEXT_KEY,
    GEO_KEY,
    REFERENCE_KEY,
    RESERVED_KEYS,
    Sample,
    checked,
    checked_iter,
    identity,
    integrity_config,
    is_remote,
    jsonable,
    normalize_path,
    sample_to_row,
)

CONFIG_FILENAME = "store_config.json"


def window_to_sample(window: GeoTileStack | GeoTile) -> Sample:
    """One window as a flat store sample — arrays, georeference, model context.

    Module-level, not a closure: optimize() pickles this for its worker
    processes. Features are dropped, being irrelevant to training and the
    one part of an anchor that will not serialize cheaply.

    Args:
        window: Window to pack. A bare GeoTile is packed as one layer named
            `"image"`.

    Returns:
        {
            "<layer>": np.ndarray,  # (band, y, x) or (time, band, y, x)
            "geo": {
                "<layer>": {...},  # encode_anchor's own dict; its header's
                                   # "array" namespace carries the band
                                   # names, times and nodata numpy loses
            },
            "model_context": {"<key>": np.ndarray | str | None},
            "reference_layer": "<layer>",  # which layer's anchor is the window's own
        }

    Raises:
        ValueError: A layer's name is one of the sample's own reserved
            fields — see `RESERVED_KEYS`.
    """
    stack = _as_stack(window)
    return _encode_sample(stack, stack.to_numpy())


def batch_to_samples(windows: Sequence[GeoTileStack | GeoTile]) -> Iterator[Sample]:
    """Several windows as store samples, their pixels read in one Dask pass.

    A generator, which `optimize()` detects and drains — one input becomes many
    samples. Windows cut from one surface share chunks, so reading a batch
    together collapses those reads. Module-level: `optimize()` pickles it.

    Args:
        windows: Windows to pack, read together. A bare GeoTile is packed as
            one layer named `"image"`.

    Yields:
        One sample per window, in the order given — same shape as
        `window_to_sample`.

    Raises:
        ValueError: A layer's name is one of the sample's own reserved fields.
    """
    stacks = [_as_stack(window) for window in windows]
    for stack, pixels in zip(stacks, read_windows(stacks), strict=True):
        yield _encode_sample(stack, pixels)


def _as_stack(window: GeoTileStack | GeoTile) -> GeoTileStack:
    """One window as a layer-keyed stack, with its layer names checked.

    Args:
        window: Window to normalize. A bare GeoTile becomes one layer named
            `"image"`.

    Returns:
        The window as a GeoTileStack.

    Raises:
        ValueError: A layer's name is one of the sample's own reserved fields.
    """
    stack = window if isinstance(window, GeoTileStack) else GeoTileStack({DEFAULT_LAYER: window})
    clashing = sorted(RESERVED_KEYS.intersection(stack))
    if clashing:
        raise ValueError(f"layer(s) {clashing} collide with a sample's reserved keys {sorted(RESERVED_KEYS)}")
    return stack


def _encode_sample(stack: GeoTileStack, pixels: dict[str, np.ndarray]) -> Sample:
    """One window's already-read pixels plus the metadata numpy loses.

    Args:
        stack: Window the pixels came from.
        pixels: That window's layers, already computed.

    Returns:
        The flat store sample — see `window_to_sample`.
    """
    sample: Sample = dict(pixels)
    sample[GEO_KEY] = {name: encode_anchor(tile.anchor) for name, tile in stack.items()}
    sample[CONTEXT_KEY] = numpy_context(stack.model_context)
    sample[REFERENCE_KEY] = stack.reference_layer
    return sample


def sample_to_window(sample: Sample) -> GeoTileStack:
    """Rebuild the window `window_to_sample` packed — for inspection, not training.

    Reading a store for training goes through `StoreDataset`, which serves
    tensors straight out of the sample. This rebuilds the xarray objects
    around them so a stored sample can be plotted or examined.

    Args:
        sample: One decoded store sample.

    Returns:
        GeoTileStack over the same grids, bands, nodata, header and reference
        layer. Its layers carry no features — `window_to_sample` does not
        store them. A store written before the reference layer was recorded
        anchors on its first layer.

    Raises:
        KeyError: `sample` carries no `"geo"` entry for one of its layers.
        ValueError: A layer's stored anchor names no bands, so its raw
            pixels cannot be put back on their grid.
    """
    geo = sample[GEO_KEY]
    layers: dict[str, GeoTile] = {}
    for name, encoded in geo.items():
        anchor = decode_anchor(encoded)
        spec = anchor.header.array
        if spec is None or spec.bands is None:
            raise ValueError(f"layer {name!r} stores no band names — its pixels can't be rebuilt")
        layers[name] = anchor.to_geotile(
            sample[name],
            bands=list(spec.bands),
            times=None if spec.times is None else list(spec.times),
            nodata=spec.nodata,
        )
    return GeoTileStack(
        layers,
        reference_layer=sample.get(REFERENCE_KEY),
        model_context=sample.get(CONTEXT_KEY) or None,
    )


class LitDataStoreConfig(TypedDict, total=False):
    """Keyword args accepted by LitDataStore() — this store's litdata.optimize() config.

    fn/inputs/output_dir/mode aren't here — those are write()'s own
    per-call args. Only the subset that actually determines whether a
    store's chunks stay decodable across writes (chunk_size, chunk_bytes,
    compression, encryption, item_loader — see utils.datastore.INTEGRITY_FIELDS)
    is locked and checked against store_config.json on construction; the
    rest (num_workers, verbose, storage_options, ...) is an execution/
    credentials knob free to differ between writes. A field left unset
    keeps optimize()'s own default, noted below.

    Args:
        input_dir: Path your files live under, downloaded in the
            background while processing. Default: None.
        weights: Per-input weight, balances work across workers. Default:
            None (even split).
        chunk_size: Max elements per chunk. Default: None.
        chunk_bytes: Max bytes per chunk. Default: None.
        align_chunking: Pack full chunks first, remainder to the last
            worker — uneven load. Default: False.
        compression: Compression algorithm over chunks. Default: None (none).
        encryption: Encryption algorithm over chunks. Default: None (none).
        num_workers: Worker count for processing. Default: None (litdata picks).
        fast_dev_run: Process only a small sub part of inputs, for a quick
            smoke run. Default: False.
        num_nodes: Node count for remote execution, lightning.ai only. Default: None.
        machine: Machine type for remote execution, lightning.ai only. Default: None.
        num_downloaders: Downloaders per worker. Default: None (litdata picks).
        num_uploaders: Uploaders per worker. Default: None (litdata picks).
        reorder_files: Reorder by file size to balance workers; False
            preserves input order. Default: True.
        reader: Reader used to read the data. Default: None (BaseReader).
        batch_size: Group inputs into batches of this length before fn()
            sees them. Default: None.
        use_checkpoint: Checkpoint progress, resumable if interrupted. Default: False.
        item_loader: Chunk item loader — sets on-disk/load format. Default: None.
        start_method: Multiprocessing start method. Default: spawn (fork inside a notebook).
        optimize_dns: Use optimized DNS resolution. Default: None.
        storage_options: Forwarded to the cloud provider's filesystem. Default:
            {}. For a hf://buckets/... path this merges onto the gateway's
            own auto-derived settings (endpoint_url, region_name, config) —
            put credentials here if not using boto3's own env vars/~/.aws/credentials.
            A missing-credentials warning fires at construction (see
            utils.datastore.warn_if_missing_aws_credentials) if none of
            storage_options/env vars/~/.aws/credentials
            look reachable — best-effort, doesn't catch an IAM instance role.
            If it was a false negative, the real failure only surfaces on
            first read/write, as a bare ValueError several frames deep in
            litdata's own S3 dependency ("Received None from
            session.get_credentials") — not a geosave-engine error, but
            that's what it means if you hit it.
        keep_data_ordered: Static per-worker item assignment; False shares
            a queue, less idle time, unordered. Default: True.
        verbose: Print optimize()'s own progress. Default: True.
        broadcast_paths: Broadcast resolved dirs across multi-node ranks;
            auto-on for {%strftime} paths. Default: False.
    """

    input_dir: str
    weights: list[int]
    chunk_size: int
    chunk_bytes: int | str
    align_chunking: bool
    compression: str
    encryption: Encryption
    num_workers: int
    fast_dev_run: bool
    num_nodes: int
    machine: str
    num_downloaders: int
    num_uploaders: int
    reorder_files: bool
    reader: BaseReader
    batch_size: int
    use_checkpoint: bool
    item_loader: BaseItemLoader
    start_method: str
    optimize_dns: bool
    storage_options: dict[str, Any]
    keep_data_ordered: bool
    verbose: bool
    broadcast_paths: bool


class LitDataStore:
    """Many samples, packed into one litdata store.

    Each write() is its own litdata.optimize() call — call it again to
    grow the store (mode="append") or replace it (mode="overwrite").
    Config is locked at construction and checked against any
    store_config.json already at path; the sample field set locks at the
    first write() and is checked on every later one. Sidecar read/write
    works the same for a remote path (s3://, gs://, r2://) as local, via
    litdata's own fs_provider — same storage_options it already reads chunks with.

    litdata itself only writes to a local path or s3://, gs://, r2:// (its
    own hardcoded provider list — not general fsspec). hf://buckets/<namespace>/
    <bucket>[/<key>] also works — a Hugging Face Storage Bucket is S3-compatible,
    so it's rewritten to an equivalent s3:// URI + gateway storage_options at
    construction (see utils.datastore.normalize_path). hf://datasets/... (a Hub dataset repo)
    is a separate, non-S3 thing and isn't writable this way — write local/s3/gs/r2
    then push the whole store with the `upload_dataset_to_hf.py` boilerplate script
    (`geosave make scripts upload_dataset_to_hf.py`) instead. Pushing to a named
    destination (HF Hub, a plain S3 bucket, ...) is workflow, not store behavior —
    deliberately not a method here.

    Args:
        path: Store root — local, s3://, gs://, r2://, or
            hf://buckets/<namespace>/<bucket>[/<key>]. Kept as given (once
            resolved), not wrapped in Path — that would corrupt a remote URI's "//".
        **config: litdata.optimize() config — see LitDataStoreConfig for the
            full field list, each field's meaning, and its optimize()
            default when left unset. Exactly one of chunk_size/chunk_bytes required.

    Raises:
        ValueError: Neither or both of chunk_size/chunk_bytes given, path
            is an unsupported/malformed URI, or a store_config.json already
            at path has a different config.

    Examples:
        >>> store = LitDataStore("data/train", chunk_size=1000)
        >>> store.write(samples)  # fn="auto" detects a plain-dict sample
        >>> len(store)
        >>> store[0]

        >>> # write straight to a Hugging Face Storage Bucket
        >>> store = LitDataStore(
        ...     "hf://buckets/my-namespace/my-bucket/train",
        ...     chunk_size=1000,
        ...     storage_options={"aws_access_key_id": "HFAK...", "aws_secret_access_key": "..."},
        ... )
    """

    def __init__(self, path: str | Path, **config: Unpack[LitDataStoreConfig]) -> None:
        if (config.get("chunk_size") is None) == (config.get("chunk_bytes") is None):
            raise ValueError("Pass exactly one of chunk_size or chunk_bytes")

        path, storage_options = normalize_path(path, config.get("storage_options"))
        if storage_options is not None:
            config["storage_options"] = storage_options

        self.path = path
        self.config: LitDataStoreConfig = config
        self._dataset: StreamingDataset | None = None
        self._provider: FsProvider | None = None

        # Cached, not re-read per call — write()/fields reuse this. A store_config.json
        # rewritten out-of-band by another process won't be picked up until re-construction,
        # same single-writer-per-instance assumption _dataset already makes.
        self._sidecar = self._read_sidecar()
        if self._sidecar is not None:
            current = jsonable(integrity_config(dict(config)))
            diff = {
                key: (self._sidecar["config"].get(key), current.get(key))
                for key in self._sidecar["config"].keys() | current.keys()
                if self._sidecar["config"].get(key) != current.get(key)
            }
            if diff:
                raise ValueError(f"{path} already has {CONFIG_FILENAME} with a different config: {diff}")

    def __repr__(self) -> str:
        """Multi-line debug summary — path, locked config, sample count, per-layer geotag.

        Opens sample 0 to read its layers and their georeference (same cost as any other read,
        cached after the first call via `_get_dataset`). Never raises —
        degrades to an inline error instead, so a broken/unreachable store
        doesn't crash a REPL/notebook's implicit repr call. storage_options
        (may carry credentials) is never shown — only INTEGRITY_FIELDS.
        """
        config = integrity_config(dict(self.config))
        config_str = ", ".join(
            f"{key}={type(value).__name__ if key in ('encryption', 'item_loader') and value is not None else value!r}"
            for key, value in config.items()
        )
        lines = [f"LitDataStore({self.path!r})", f"  config:  {config_str}"]

        try:
            n = len(self)
            sample = self[0]
        except Exception as e:  # repr must never raise — degrade instead
            lines.append(f"  <no data — {e}>")
            return "\n".join(lines)

        lines.append(f"  samples: {n}")
        lines.append(f"  fields:  {tuple(sample)!r}")
        lines.append("  layers:")
        for layer_name, encoded in sample.get(GEO_KEY, {}).items():
            spec = ArraySpec.decode((encoded.get("header") or {}).get(ArraySpec.NAMESPACE) or {})
            when = f"{spec.times[0]}–{spec.times[-1]}" if spec.times else None
            lines.append(
                f"    {layer_name} — bands={spec.bands}, times={when}, nodata={spec.nodata}"
            )
        return "\n".join(lines)

    def write(
        self,
        samples: Sequence[Any],
        fn: Callable[[Any], Sample] | Callable[[Any], Iterator[Sample]] | Literal["auto"] = "auto",
        mode: Literal["append", "overwrite"] | None = None,
    ) -> Path | str:
        """Write samples via one litdata.optimize() call, using this store's locked config.

        Args:
            samples: Indexable sequence to write — a plain-dict sample
                sequence, or anything fn knows how to convert. Not a bare
                dict — that's one sample's fields, not a sequence of
                samples; optimize() would iterate its keys instead.
            fn: "auto" detects samples[0]'s type — a plain dict passes
                through, a GeoTileStack/GeoTile goes through
                `window_to_sample`, anything else raises. A custom fn must be
                a module-level function (not a lambda/closure), since
                optimize() pickles it for its worker processes. A generator
                fn expands one input into several samples — see
                `batch_to_samples`; litdata cannot checkpoint one, so
                `use_checkpoint` is rejected alongside it.
            mode: None raises if path already holds a store (litdata's own
                guard). "append" grows it, "overwrite" replaces it.

        Returns:
            Store root, as given at construction.

        Call this under `if __name__ == "__main__":` in a script. The default
        `start_method` is spawn, so each worker re-imports the calling module —
        a bare module-level `write()` re-enters itself in every child and hangs.

        Raises:
            ValueError: samples is empty, fn="auto" can't handle samples[0]'s
                type, a sample's field set doesn't match this store's locked
                one, or samples[0]'s built sample isn't picklable (same
                requirement as fn — optimize() ships items to worker processes).
        """
        if not samples:
            raise ValueError("No samples given — nothing to write")
        if isinstance(samples, (dict, str, bytes)):
            raise ValueError("samples must be a sequence of samples (e.g. list, tuple), not a single dict or string")
        if not isinstance(samples, Sequence):
            samples = list(samples)  # Convert generators/sets if allowed, or raise TypeError

        if fn == "auto":
            first = samples[0]
            if isinstance(first, dict):
                fn = identity
            elif isinstance(first, (GeoTileStack, GeoTile)):
                fn = window_to_sample
            else:
                raise ValueError(f"No auto serializer for {type(first).__name__} — pass fn explicitly")

        # Ensure fn itself can be pickled for worker processes
        try:
            pickle.dumps(fn)
        except Exception as e:
            raise ValueError(f"The serializer function `fn` ({fn}) is not picklable for worker processes: {e}") from e

        expands = inspect.isgeneratorfunction(fn)
        if expands and self.config.get("use_checkpoint"):
            raise ValueError("litdata cannot checkpoint a generator fn — drop use_checkpoint or fn's yield")

        # 4. Validate fields and sample picklability
        if self._sidecar is not None:
            expected_fields = frozenset(self._sidecar["fields"])
        else:
            produced = fn(samples[0])
            first_sample = next(iter(produced)) if expands else produced

            if not isinstance(first_sample, Mapping):
                raise TypeError(
                    f"Expected sample serializer `fn` to return a dict/Mapping of fields, but got {type(first_sample).__name__}"
                )

            try:
                pickle.dumps(first_sample)
            except Exception as e:
                raise ValueError(
                    f"samples[0]'s built sample isn't picklable, optimize() ships it to worker processes: {e}"
                ) from e

            expected_fields = frozenset(first_sample)

        # cast: isgeneratorfunction sees through partial, so litdata still detects an expanding fn
        guard = cast("Any", checked_iter if expands else checked)
        optimize(
            # bind fn/expected_fields now — optimize() calls this with one item at a time
            fn=functools.partial(guard, fn=fn, expected_fields=expected_fields),
            inputs=samples,
            output_dir=str(self.path),
            mode=mode,
            **self.config,
        )
        self._dataset = None  # stale after a write, StreamingDataset re-opens lazily
        self._write_sidecar(expected_fields)
        return Path(self.path) if not is_remote(self.path) else self.path

    def to_pandas(self) -> pd.DataFrame:
        """Write this store's metadata to a pandas DataFrame, one row per sample.

        Drops payload fields (any array/tensor value — the pixel data,
        looked up by `self._get_dataset()`'s own decode) and flattens the
        rest (e.g. "geo" -> "geo_<layer>_<key>" columns), so a layer's
        recorded bands or nodata end up directly queryable.
        `"index"` column points back at this store's row position.

        Returns:
            `pd.DataFrame`, one row per sample.
        """
        rows = [sample_to_row(self[i], i) for i in range(len(self))]
        return pd.json_normalize(rows, sep="_")

    def to_parquet(self, path: str | Path) -> None:
        """Write this store's metadata to a parquet file, one row per sample.
        
        Args:
            path: Output `.parquet` file path.
        """
        self.to_pandas().to_parquet(path)

    def __len__(self) -> int:
        """Sample count, read from the store's index — no data load."""
        return len(self._get_dataset())

    def __getitem__(self, index: int) -> Sample:
        """Sample at position index, decoded back to a plain dict."""
        return self._get_dataset()[index]

    @property
    def fields(self) -> tuple[str, ...]:
        """Top-level field names every sample in this store carries.

        Locked from the first write()'s first sample.

        Raises:
            ValueError: Nothing written yet — no store_config.json.
        """
        if self._sidecar is None:
            raise ValueError(f"{self.path} has no {CONFIG_FILENAME} yet — call write() first")
        return tuple(self._sidecar["fields"])

    # ------------------------------------------------------------------
    # Sidecar / dataset handle
    # ------------------------------------------------------------------

    def _get_dataset(self) -> StreamingDataset:
        if self._dataset is None:
            self._dataset = StreamingDataset(str(self.path))
        return self._dataset

    def _get_provider(self) -> FsProvider:
        if self._provider is None:
            self._provider = _get_fs_provider(str(self.path), storage_options=self.config.get("storage_options"))
        return self._provider

    def _remote_sidecar_path(self) -> str:
        return f"{str(self.path).rstrip('/')}/{CONFIG_FILENAME}"

    def _read_sidecar(self) -> dict[str, Any] | None:
        """Read store_config.json at this store's path, local or remote. None if missing."""
        if is_remote(self.path):
            provider = self._get_provider()
            remote_path = self._remote_sidecar_path()
            if not provider.exists(remote_path):
                return None
            with tempfile.TemporaryDirectory() as tmp:
                local_path = Path(tmp) / CONFIG_FILENAME
                provider.download_file(remote_path, str(local_path))
                return json.loads(local_path.read_text())

        sidecar = Path(self.path) / CONFIG_FILENAME
        if not sidecar.exists():
            return None
        return json.loads(sidecar.read_text())

    def _write_sidecar(self, fields: frozenset[str]) -> None:
        """Write store_config.json — locked config (integrity-relevant fields
        only, see INTEGRITY_FIELDS) + sample field set. Works local or remote,
        updates the cached sidecar dict (self._sidecar) once written.
        """
        current = jsonable(integrity_config(dict(self.config)))
        payload_dict = {"config": current, "fields": sorted(fields)}
        payload = json.dumps(payload_dict)

        if is_remote(self.path):
            with tempfile.TemporaryDirectory() as tmp:
                local_path = Path(tmp) / CONFIG_FILENAME
                local_path.write_text(payload)
                self._get_provider().upload_file(str(local_path), self._remote_sidecar_path())
        else:
            sidecar = Path(self.path) / CONFIG_FILENAME
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(payload)

        self._sidecar = payload_dict
