"""StoreDataset: PyTorch dataset over a SampleStore's packed samples.

Inherits litdata's StreamingDataset directly. No method override —
band-select and dtype-cast run as `transform` functions litdata applies
after its own decode, so both `__getitem__` and iteration (`__next__`
calls `__getitem__` internally) pick them up for free.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from litdata import StreamingDataset
from litdata.streaming.item_loader import BaseItemLoader
from litdata.streaming.resolver import Dir
from litdata.streaming.serializers import Serializer
from litdata.utilities.encryption import Encryption

from geosave_engine.geodata.utils.datastore import normalize_path, sample_to_row

LayerName = str


class StoreDataset(StreamingDataset):
    """PyTorch dataset over a SampleStore's packed samples.

    Args:
        path: Store root, as given to SampleStore() — local, s3://, gs://,
            r2://, or hf://buckets/<namespace>/<bucket>[/<key>] (rewritten
            to its equivalent s3:// URI + gateway storage_options, same as
            SampleStore does).
        sel_bands: Layer name to band names to keep. Default keeps all
            bands the layer carries. Needs GeoTag.bands recorded at write
            time — raises if a selected layer has none.
        dtype_override: Layer name to torch dtype to cast that layer's
            tensor to. Default keeps the stored dtype.
        cache_dir: Local cache folder for downloaded chunks. Default:
            litdata's own cache directory.
        item_loader: Chunk item loader — must match what the store was
            written with. Default: None (litdata's PyTreeLoader), right
            for a plain-dict/GeoStack.to_numpy() store.
        shuffle: Shuffle sample order. Default: False.
        drop_last: Drop the last incomplete batch in a distributed setting.
            Default: None (litdata picks True if distributed, else False).
        seed: Shuffle random seed. Default: 42.
        serializers: Override litdata's own decoders, keyed by format name.
            Default: None (litdata's built-ins) — right unless the store
            was written with custom serializers.
        max_cache_size: Max on-disk cache size. Default: "100GB".
        subsample: Fraction of the store to randomly sample. Default: 1.0 (all).
        encryption: Decryption for an encrypted store — must match what the
            store was written with. Default: None (not encrypted).
        storage_options: Cloud provider filesystem options (credentials,
            ...). Default: None. Merged onto hf://buckets/...'s own
            gateway settings if path is one.
        session_options: S3 session options. Default: None.
        max_pre_download: Chunks to pre-download ahead of the reader.
            Default: 2.
        index_path: Path to a Parquet dataset's index.json. Default: None
            (not a Parquet dataset).
        force_override_state_dict: Let local args override a loaded
            checkpoint's state. Default: False.
    """

    def __init__(
        self,
        path: str | Path,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
        *,
        cache_dir: "str | Dir | None" = None,
        item_loader: BaseItemLoader | None = None,
        shuffle: bool = False,
        drop_last: bool | None = None,
        seed: int = 42,
        serializers: dict[str, Serializer] | None = None,
        max_cache_size: int | str = "100GB",
        subsample: float = 1.0,
        encryption: Encryption | None = None,
        storage_options: dict[str, Any] | None = None,
        session_options: dict[str, Any] | None = None,
        max_pre_download: int = 2,
        index_path: str | None = None,
        force_override_state_dict: bool = False,
    ) -> None:
        path, storage_options = normalize_path(path, storage_options)
        transform: list[Callable[[dict[str, Any]], dict[str, Any]]] = [
            functools.partial(_select_bands, sel_bands=sel_bands),
            functools.partial(_cast_dtypes, dtype_override=dtype_override),
        ]
        super().__init__(
            input_dir=str(path),
            cache_dir=cache_dir,
            item_loader=item_loader,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            serializers=serializers,
            max_cache_size=max_cache_size,
            subsample=subsample,
            encryption=encryption,
            storage_options=storage_options,
            session_options=session_options,
            max_pre_download=max_pre_download,
            index_path=index_path,
            force_override_state_dict=force_override_state_dict,
            transform=transform,
        )

    def to_pandas(self) -> pd.DataFrame:
        """Snapshot every sample's non-array fields into one table.

        Decodes every sample in full to get there — litdata has no
        metadata-only read, same cost as SampleStore.to_parquet()'s own loop.

        Returns:
            DataFrame, one row per sample, flattened (e.g. "geotags_<layer>_<key>").
        """
        rows = [sample_to_row(self[i], i) for i in range(len(self))]
        return pd.json_normalize(rows, sep="_")


def _select_bands(
    sample: dict[str, Any],
    sel_bands: dict[LayerName, list[str]] | None,
) -> dict[str, Any]:
    """Keep only sel_bands' band names per layer; other layers untouched.

    Module-level, not a closure — StoreDataset.__init__ binds sel_bands via
    functools.partial so the transform stays picklable for DataLoader workers.

    Args:
        sample: Decoded sample dict, layer name to band-stacked array, plus
            "geotags" (layer name to that layer's dumped GeoTag).
        sel_bands: Layer name to band names to keep. None is a no-op.

    Returns:
        Sample dict, same keys, selected layers' arrays sliced to sel_bands.

    Raises:
        ValueError: A selected layer has no recorded GeoTag.bands, or asks
            for a band name that layer doesn't carry.
    """
    if not sel_bands:
        return sample

    for layer_name, bands in sel_bands.items():
        array = sample[layer_name]
        layer_bands = sample["geotags"][layer_name].get("bands")
        if layer_bands is None:
            raise ValueError(f"{layer_name!r} has no band names stored — sel_bands needs GeoTag.bands")
        missing = set(bands) - set(layer_bands)
        if missing:
            raise ValueError(f"{layer_name!r} band(s) {sorted(missing)} not in stored bands {layer_bands}")

        indices = [layer_bands.index(b) for b in bands]
        band_axis = 1 if array.ndim == 4 else 0  # (time, band, y, x) vs (band, y, x)
        sample[layer_name] = np.take(array, indices, axis=band_axis)

    return sample


def _cast_dtypes(
    sample: dict[str, Any],
    dtype_override: dict[LayerName, torch.dtype] | None,
) -> dict[str, Any]:
    """Convert array fields to tensors, casting dtype_override's layers.

    Module-level, not a closure — same picklability reason as _select_bands.
    Runs on every sample regardless of dtype_override — numpy -> tensor
    conversion isn't conditional, only the cast itself is.

    Args:
        sample: Decoded sample dict, layer name to numpy array.
        dtype_override: Layer name to torch dtype to cast to. None keeps
            the stored dtype.

    Returns:
        Sample dict, same keys, array fields converted to tensors.
    """
    dtype_override = dtype_override or {}

    for key, value in sample.items():
        if not isinstance(value, np.ndarray):
            continue
        # litdata decodes straight from a read-only buffer (np.frombuffer) — copy so the
        # tensor is writable, not a view a caller could mutate into undefined behavior.
        tensor = torch.from_numpy(value.copy())
        dtype = dtype_override.get(key)
        sample[key] = tensor.to(dtype) if dtype is not None else tensor

    return sample
