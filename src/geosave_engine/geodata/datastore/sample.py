"""SampleStore: many samples packed into one litdata store.

Thin wrapper around litdata's optimize()/StreamingDataset — for a plain
dict sample it stays domain-blind; fn="auto" additionally knows a GeoStack
sequence, via GeoStack.to_numpy() itself (arrays + "geobox"/"geotags").
"""
from __future__ import annotations

import functools
import json
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypedDict

import pandas as pd
from litdata import optimize
from litdata.processing.readers import BaseReader
from litdata.streaming import StreamingDataset
from litdata.streaming.item_loader import BaseItemLoader
from litdata.utilities.encryption import Encryption
from typing_extensions import Unpack

from geosave_engine.geodata.spatial.stack import GeoStack
from geosave_engine.geodata.utils.datastore import (
    Sample,
    checked,
    identity,
    integrity_config,
    is_remote,
    jsonable,
    normalize_path,
    sample_to_row,
)

if TYPE_CHECKING:
    from concurrent.futures import Future

    from huggingface_hub import CommitInfo

CONFIG_FILENAME = "store_config.json"


class SampleStoreConfig(TypedDict, total=False):
    """Keyword args accepted by SampleStore() — this store's litdata.optimize() config.

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


class UploadToHfKwargs(TypedDict, total=False):
    """Extra keyword args forwarded to huggingface_hub's HfApi.upload_folder().

    Args:
        path_in_repo: Target directory in the repo. Default: None (repo root).
        commit_message: Commit message. Default: None (auto-generated).
        commit_description: Commit description. Default: None.
        revision: Branch to commit to. Default: None (repo's default branch).
        create_pr: Open a PR instead of committing directly. Default: None (False).
        parent_commit: Parent commit SHA to base this commit on. Default:
            None (HEAD of revision).
        allow_patterns: Only upload files matching these glob(s). Default:
            None (no filter).
        ignore_patterns: Skip files matching these glob(s). Default: None (no filter).
        delete_patterns: Delete repo files matching these glob(s) that
            aren't in folder_path. Default: None (no deletion).
    """

    path_in_repo: str | None
    commit_message: str
    commit_description: str
    revision: str
    create_pr: bool
    parent_commit: str
    allow_patterns: list[str] | str
    ignore_patterns: list[str] | str
    delete_patterns: list[str] | str


class SampleStore:
    """Many samples, packed into one litdata store.

    Each write() is its own litdata.optimize() call — call it again to
    grow the store (mode="append") or replace it (mode="overwrite").
    Config is locked at construction and checked against any
    store_config.json already at path; the sample field set locks at the
    first write() and is checked on every later one. Sidecar checks only
    run for a local path — a remote path (s3://, ...) skips them.

    litdata itself only writes to a local path or s3://, gs://, r2:// (its
    own hardcoded provider list — not general fsspec). hf://buckets/<namespace>/
    <bucket>[/<key>] also works — a Hugging Face Storage Bucket is S3-compatible,
    so it's rewritten to an equivalent s3:// URI + gateway storage_options at
    construction (see utils.datastore.normalize_path). hf://datasets/... (a Hub dataset repo)
    is a separate, non-S3 thing and isn't writable this way — write
    local/s3/gs/r2 then push the whole store with upload_to_hf() instead;
    same idea for a plain S3 bucket via upload_to_s3().

    Args:
        path: Store root — local, s3://, gs://, r2://, or
            hf://buckets/<namespace>/<bucket>[/<key>]. Kept as given (once
            resolved), not wrapped in Path — that would corrupt a remote URI's "//".
        **config: litdata.optimize() config — see SampleStoreConfig for the
            full field list, each field's meaning, and its optimize()
            default when left unset. Exactly one of chunk_size/chunk_bytes required.

    Raises:
        ValueError: Neither or both of chunk_size/chunk_bytes given, path
            is an unsupported/malformed URI, or a store_config.json already
            at path has a different config.

    Examples:
        >>> store = SampleStore("data/train", chunk_size=1000)
        >>> store.write(stacks)  # fn="auto" detects the GeoStack sequence
        >>> len(store)
        >>> store[0]

        >>> # write straight to a Hugging Face Storage Bucket
        >>> store = SampleStore(
        ...     "hf://buckets/my-namespace/my-bucket/train",
        ...     chunk_size=1000,
        ...     storage_options={"aws_access_key_id": "HFAK...", "aws_secret_access_key": "..."},
        ... )

        >>> # push to Hugging Face Hub after a local/s3/gs/r2 write
        >>> store.upload_to_hf("org/dataset-name")
    """

    def __init__(self, path: str | Path, **config: Unpack[SampleStoreConfig]) -> None:
        if (config.get("chunk_size") is None) == (config.get("chunk_bytes") is None):
            raise ValueError("Pass exactly one of chunk_size or chunk_bytes")

        path, storage_options = normalize_path(path, config.get("storage_options"))
        if storage_options is not None:
            config["storage_options"] = storage_options

        self.path = path
        self.config: SampleStoreConfig = config
        self._dataset: StreamingDataset | None = None

        existing = self._read_sidecar()
        if existing is not None:
            current = jsonable(integrity_config(dict(config)))
            diff = {
                key: (existing["config"].get(key), current.get(key))
                for key in existing["config"].keys() | current.keys()
                if existing["config"].get(key) != current.get(key)
            }
            if diff:
                raise ValueError(f"{path} already has {CONFIG_FILENAME} with a different config: {diff}")

    def __repr__(self) -> str:
        """Multi-line debug summary — path, locked config, sample count, per-layer geotag.

        Opens sample 0 to read layers/geotags (same cost as any other read,
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
        lines = [f"SampleStore({self.path!r})", f"  config:  {config_str}"]

        try:
            n = len(self)
            sample = self[0]
        except Exception as e:  # repr must never raise — degrade instead
            lines.append(f"  <no data — {e}>")
            return "\n".join(lines)

        lines.append(f"  samples: {n}")
        lines.append(f"  fields:  {tuple(sample)!r}")
        lines.append("  layers:")
        for layer_name, geotag in sample.get("geotags", {}).items():
            bands = geotag.get("bands")
            lines.append(f"    {layer_name} — bands={tuple(bands) if bands else None}, datetime={geotag.get('datetime')}")
        return "\n".join(lines)

    def write(
        self,
        samples: Sequence[Any],
        fn: Callable[[Any], Sample] | Literal["auto"] = "auto",
        mode: Literal["append", "overwrite"] | None = None,
    ) -> Path | str:
        """Write samples via one litdata.optimize() call, using this store's locked config.

        Args:
            samples: Indexable sequence to write — a GeoStack sequence, a
                plain-dict sample sequence, or anything fn knows how to
                convert. Not a bare dict — that's one sample's fields, not
                a sequence of samples; optimize() would iterate its keys instead.
            fn: "auto" detects samples[0]'s type (GeoStack -> to_numpy() +
                per-layer GeoAnchor; dict -> passthrough), raises for
                anything else. A custom fn must be a module-level function
                (not a lambda/closure) — optimize() pickles it for its
                worker processes.
            mode: None raises if path already holds a store (litdata's own
                guard). "append" grows it, "overwrite" replaces it.

        Returns:
            Store root, as given at construction.

        Raises:
            ValueError: samples is empty, fn="auto" can't handle samples[0]'s
                type, a sample's field set doesn't match this store's locked
                one, or samples[0]'s built sample isn't picklable (same
                requirement as fn — optimize() ships items to worker processes).
        """
        if not samples:
            raise ValueError("No samples given — nothing to write")
        if isinstance(samples, dict):
            raise ValueError("samples must be a sequence of samples, not a single dict — wrap it in a list")

        if fn == "auto":
            first = samples[0]
            if isinstance(first, GeoStack):
                fn = GeoStack.to_numpy
            elif isinstance(first, dict):
                fn = identity
            else:
                raise ValueError(f"no auto serializer for {type(first).__name__} — pass fn explicitly")

        existing = self._read_sidecar()
        if existing is not None:
            expected_fields = frozenset(existing["fields"])
        else:
            first_sample = fn(samples[0])
            try:
                pickle.dumps(first_sample)
            except Exception as e:
                raise ValueError(
                    f"samples[0]'s built sample isn't picklable, optimize() ships it to worker processes: {e}"
                ) from e
            expected_fields = frozenset(first_sample)

        optimize(
            # bind fn/expected_fields now — optimize() calls this with one item at a time
            fn=functools.partial(checked, fn=fn, expected_fields=expected_fields),
            inputs=samples,
            output_dir=str(self.path),
            mode=mode,
            **self.config,
        )
        self._dataset = None  # stale after a write, StreamingDataset re-opens lazily
        self._write_sidecar(expected_fields)
        return Path(self.path) if not is_remote(self.path) else self.path

    def upload_to_hf(
        self,
        repo_id: str,
        repo_type: str = "dataset",
        token: str | None = None,
        create_parquet: bool = False,
        **kwargs: Unpack[UploadToHfKwargs],
    ) -> "CommitInfo | Future[CommitInfo]":
        """Push this store to a Hugging Face Hub dataset repo.

        Thin wrapper over huggingface_hub's own upload — litdata has no
        push mechanism of its own. path must be local — HfApi uploads a
        local folder, not a remote-to-remote transfer.

        Args:
            repo_id: Target HF Hub dataset repo, e.g. "org/dataset-name".
            repo_type: HF Hub repo type.
            token: HF Hub auth token. Default reads the HF_TOKEN env var,
                then huggingface_hub's own cached login (`hf auth login`).
            create_parquet: Also push to_parquet()'s manifest as its own
                repo-root file, named "<store folder name>.parquet" (e.g.
                "train.parquet") — matches HF's own split-detection
                convention and won't collide if several stores' folders
                (train/val/test) end up under one repo. Built in a temp
                dir, not inside path, so it's independent of wherever
                kwargs["path_in_repo"] nests the store's own chunk files.
            **kwargs: Forwarded to HfApi.upload_folder() — see
                UploadToHfKwargs for the full field list.

        Returns:
            CommitInfo, or a Future of one if run_as_future=True.
        """
        from huggingface_hub import HfApi, get_token

        token = token or get_token()
        if token is None:
            warnings.warn(
                "No HF token found (HF_TOKEN env var unset, not logged in via `hf auth login`) "
                "— upload_to_hf will fail unless repo_id is a public repo you don't need auth for."
            )
        api = HfApi(token=token)

        if create_parquet:
            with tempfile.TemporaryDirectory() as tmp:
                manifest_path = Path(tmp) / f"{Path(self.path).name}.parquet"
                self.to_parquet(manifest_path)
                api.upload_file(
                    path_or_fileobj=str(manifest_path),
                    path_in_repo=manifest_path.name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )

        return api.upload_folder(
            folder_path=str(self.path),
            repo_id=repo_id,
            repo_type=repo_type,
            **kwargs,
        )

    def upload_to_s3(self, bucket: str, prefix: str | None = None) -> None:
        """Push this store to an S3 bucket.

        Thin wrapper over boto3's own upload — litdata has no push mechanism
        of its own. path must be local — boto3 uploads a local folder, not
        a remote-to-remote transfer.

        Args:
            bucket: Target S3 bucket name.
            prefix: Optional key prefix under the bucket.
        """
        pass

    def to_parquet(self, path: str | Path) -> Path:
        """Write this store's metadata to one Parquet file, one row per sample.

        Drops payload fields (any array/tensor value — the pixel data,
        looked up by `self._get_dataset()`'s own decode) and flattens the
        rest (e.g. "geotags" -> "geotags_<layer>_<key>" columns), so a
        custom geotag key like cloud_cover ends up directly queryable.
        `"index"` column points back at this store's row position.

        Args:
            path: Output `.parquet` file path.

        Returns:
            `path`, as given.
        """
        rows = [sample_to_row(self[i], i) for i in range(len(self))]
        path = Path(path)
        pd.json_normalize(rows, sep="_").to_parquet(path)
        return path

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
        existing = self._read_sidecar()
        if existing is None:
            raise ValueError(f"{self.path} has no {CONFIG_FILENAME} yet — call write() first")
        return tuple(existing["fields"])

    # ------------------------------------------------------------------
    # Sidecar / dataset handle
    # ------------------------------------------------------------------

    def _get_dataset(self) -> StreamingDataset:
        if self._dataset is None:
            self._dataset = StreamingDataset(str(self.path))
        return self._dataset

    def _read_sidecar(self) -> dict[str, Any] | None:
        """Read store_config.json at this store's path. None if missing or path is remote."""
        if is_remote(self.path):
            return None
        sidecar = Path(self.path) / CONFIG_FILENAME
        if not sidecar.exists():
            return None
        return json.loads(sidecar.read_text())

    def _write_sidecar(self, fields: frozenset[str]) -> None:
        """Write store_config.json — locked config (integrity-relevant fields
        only, see INTEGRITY_FIELDS) + sample field set. No-op for a remote path.
        """
        if is_remote(self.path):
            return
        sidecar = Path(self.path) / CONFIG_FILENAME
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        current = jsonable(integrity_config(dict(self.config)))
        sidecar.write_text(json.dumps({"config": current, "fields": sorted(fields)}))
