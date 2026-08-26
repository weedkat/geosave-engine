"""StoreDataset: PyTorch dataset over a LitDataStore's packed samples.

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

from geosave_engine.geodata.extensions import ArraySpec
from geosave_engine.geodata.spatial import TensorSample, decode_anchor, tensor_context
from geosave_engine.geodata.utils.datastore import CONTEXT_KEY, GEO_KEY, REFERENCE_KEY, normalize_path, sample_to_row

LayerName = str


class StoreDataset(StreamingDataset):
    """PyTorch dataset over a LitDataStore's packed samples.

    Serves the same `TensorSample` shape as `GeoPipeline.ingest_to_tensor`
    and `StackDataset`, so a model reads either path unchanged. Collate with
    `geosave_engine.geodata.datasets.stack_samples` — a GeoAnchor is not a
    tensor, and torch's default collate would shred it into struct-of-lists.

    Args:
        path: Store root, as given to LitDataStore() — local, s3://, gs://,
            r2://, or hf://buckets/<namespace>/<bucket>[/<key>] (rewritten
            to its equivalent s3:// URI + gateway storage_options, same as
            LitDataStore does).
        sel_bands: Layer name to band names to keep. Default keeps all
            bands the layer carries. Needs band names recorded at write
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
        transform: list[Callable[[dict[str, Any]], Any]] = [
            functools.partial(_select_bands, sel_bands=sel_bands),
            functools.partial(_cast_dtypes, dtype_override=dtype_override),
            _to_sample,
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
        metadata-only read, same cost as LitDataStore.to_parquet()'s own loop.

        Returns:
            DataFrame, one row per sample, flattened (e.g. "geo_<layer>_<key>").
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
            "geo" (layer name to that layer's own encoded anchor).
        sel_bands: Layer name to band names to keep. None is a no-op.

    Returns:
        Sample dict, same keys, selected layers' arrays sliced to sel_bands.

    Raises:
        ValueError: A selected layer has no recorded band names, asks for
            a band name that layer doesn't carry, or the band axis can't
            be located from the array's own shape.
    """
    if not sel_bands:
        return sample

    for layer_name, bands in sel_bands.items():
        array = sample[layer_name]
        encoded = sample[GEO_KEY][layer_name]
        layer_bands = ArraySpec.decode((encoded.get("header") or {}).get(ArraySpec.NAMESPACE) or {}).bands
        if layer_bands is None:
            raise ValueError(f"{layer_name!r} has no band names stored — sel_bands needs the layer's own bands")
        missing = set(bands) - set(layer_bands)
        if missing:
            raise ValueError(f"{layer_name!r} band(s) {sorted(missing)} not in stored bands {layer_bands}")

        indices = [layer_bands.index(b) for b in bands]
        # band axis is whichever of (band,y,x)/(time,band,y,x)'s first two axes actually holds len(layer_bands) entries
        candidates = [axis for axis in (0, 1) if axis < array.ndim and array.shape[axis] == len(layer_bands)]
        if len(candidates) != 1:
            raise ValueError(f"{layer_name!r} array shape {array.shape} doesn't unambiguously locate its {len(layer_bands)} band(s)")
        sample[layer_name] = np.take(array, indices, axis=candidates[0])

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


def _to_sample(sample: dict[str, Any]) -> TensorSample:
    """Reshape one decoded store sample into the TensorSample every path serves.

    Module-level, not a closure — same picklability reason as `_select_bands`.
    Runs last, after band selection and tensor conversion have done their work
    on the flat layer keys.

    Args:
        sample: Decoded sample dict — layer name to tensor, plus the
            reserved "geo", "model_context" and "reference_layer" fields.

    Returns:
        {
            "layers": {"<layer>": torch.Tensor},
            "anchor": GeoAnchor,
            "model_context": {"<key>": torch.Tensor | str | None},
        }
        The anchor is the stored reference layer's — the one whose anchor was
        the window's own identity. A store written before that was recorded
        falls back to its first layer.

    Raises:
        KeyError: `sample` carries no "geo" entry — not a window-shaped store.
    """
    geo = sample.pop(GEO_KEY)
    context = sample.pop(CONTEXT_KEY, None)
    reference = sample.pop(REFERENCE_KEY, None)
    anchored = geo.get(reference) if reference is not None else None
    return {
        "layers": {name: value for name, value in sample.items()},
        "anchor": decode_anchor(anchored if anchored is not None else next(iter(geo.values()))),
        "model_context": tensor_context(context or None),
    }
