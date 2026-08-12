"""Path/config plumbing for SampleStore — kept out of sample.py so that file is just the class.

Nothing here is public API; SampleStore is the only caller.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

Sample = dict[str, Any]

# litdata's own index.json persists exactly these under its "config" key —
# they're what actually determines whether a store's chunks stay decodable
# across writes. Everything else in SampleStoreConfig (num_workers, verbose,
# storage_options, ...) is an execution/credentials knob that can freely
# differ between writes, so it's neither locked nor written to the sidecar.
INTEGRITY_FIELDS = frozenset({"chunk_size", "chunk_bytes", "compression", "encryption", "item_loader"})

HF_BUCKET_PREFIX = "hf://buckets/"
WRITABLE_SCHEMES = ("s3", "gs", "r2")  # litdata's own _SUPPORTED_PROVIDERS — all it can write/read remotely


def is_remote(path: str | Path) -> bool:
    """True if path is a remote URI (s3://, gs://, r2://), not a local path."""
    return "://" in str(path)


def jsonable(config: dict[str, Any]) -> dict[str, Any]:
    """Best-effort JSON-safe form of a config dict, for the sidecar drift-check.

    Recurses through nested values too (e.g. an object buried in
    storage_options), not just top-level ones — anything json can't
    handle natively falls back to str(), same as json.dumps's own default=.
    """
    return json.loads(json.dumps(config, default=str))


def integrity_config(config: dict[str, Any]) -> dict[str, Any]:
    """Subset of config that's locked/persisted — see INTEGRITY_FIELDS."""
    return {k: v for k, v in config.items() if k in INTEGRITY_FIELDS}


def identity(x: Any) -> Any:
    return x


def checked(item: Any, fn: Callable[[Any], Sample], expected_fields: frozenset[str]) -> Sample:
    """Run fn, then enforce every sample shares the store's locked field set."""
    sample = fn(item)
    got_fields = frozenset(sample)
    if got_fields != expected_fields:
        raise ValueError(
            f"sample fields {sorted(got_fields)} don't match this store's locked fields {sorted(expected_fields)}"
        )
    return sample


def parse_hf_bucket_path(path: str) -> tuple[str, str]:
    """Split hf://buckets/<namespace>/<bucket>[/<key>] into (namespace, equivalent s3:// URI).

    Args:
        path: A path already confirmed to start with HF_BUCKET_PREFIX.

    Returns:
        (namespace, "s3://<bucket>" or "s3://<bucket>/<key>").

    Raises:
        ValueError: Missing namespace or bucket segment.
    """
    namespace, _, rest = path.removeprefix(HF_BUCKET_PREFIX).partition("/")
    bucket, _, key = rest.partition("/")
    if not namespace or not bucket:
        raise ValueError(f"{path!r} must be {HF_BUCKET_PREFIX}<namespace>/<bucket>[/<key>]")
    return namespace, f"s3://{bucket}/{key}" if key else f"s3://{bucket}"


def hf_bucket_storage_options(namespace: str) -> dict[str, Any]:
    """Gateway settings a Hugging Face Storage Bucket needs on top of normal S3 credentials.

    See https://huggingface.co/docs/hub/en/storage-buckets-s3. Credentials
    (aws_access_key_id/aws_secret_access_key) aren't set here — boto3's own
    chain picks those up from storage_options, env vars, or ~/.aws/credentials,
    same as any other S3 bucket; generate them via an HF access token's
    "Generate S3 credentials" action.
    """
    from botocore.config import Config

    return {
        "endpoint_url": f"https://s3.hf.co/{namespace}",
        "region_name": "us-east-1",  # gateway is single-region, required regardless
        "config": Config(
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    }


def normalize_path(
    path: str | Path, storage_options: dict[str, Any] | None
) -> tuple[str | Path, dict[str, Any] | None]:
    """Resolve path to what litdata actually accepts, translating a Storage Bucket URI.

    hf://buckets/<namespace>/<bucket>[/<key>] becomes s3://<bucket>[/<key>],
    with storage_options merged onto the gateway's own settings (caller's
    values win on conflict). A local path or litdata's own s3/gs/r2 schemes
    pass through unchanged.

    Args:
        path: As given to SampleStore().
        storage_options: As given in SampleStoreConfig, if any.

    Returns:
        (path litdata should use, storage_options litdata should use).

    Raises:
        ValueError: hf://datasets/... (not writable directly — see
            SampleStore.upload_to_hf) or any other unsupported scheme.
    """
    if not isinstance(path, str) or "://" not in path:
        return path, storage_options

    if path.startswith(HF_BUCKET_PREFIX):
        namespace, s3_path = parse_hf_bucket_path(path)
        return s3_path, {**hf_bucket_storage_options(namespace), **(storage_options or {})}

    scheme = path.split("://", 1)[0]
    if scheme == "hf":
        raise ValueError(
            f"{path!r} — hf://datasets/... isn't writable directly (litdata only writes "
            f"local or {WRITABLE_SCHEMES}); write there then call upload_to_hf(), or use "
            f"{HF_BUCKET_PREFIX}<namespace>/<bucket>[/<key>] for a Storage Bucket"
        )
    if scheme not in WRITABLE_SCHEMES:
        raise ValueError(f"{path!r} — unsupported scheme, must be local or one of {WRITABLE_SCHEMES}")
    return path, storage_options
