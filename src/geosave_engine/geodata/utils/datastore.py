"""Path/config plumbing for SampleStore — kept out of sample.py so that file is just the class.

Nothing here is public API; SampleStore is the only caller.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

Sample = dict[str, Any]

# litdata's own index.json persists exactly these under its "config" key —
# they're what actually determines whether a store's chunks stay decodable
# across writes. Everything else in SampleStoreConfig (num_workers, verbose,
# storage_options, ...) is an execution/credentials knob that can freely
# differ between writes, so it's neither locked nor written to the sidecar.
INTEGRITY_FIELDS = frozenset({"chunk_size", "chunk_bytes", "compression", "encryption", "item_loader"})

HF_BUCKET_PREFIX = "hf://buckets/"
WRITABLE_SCHEMES = ("s3", "gs", "r2")  # litdata's own _SUPPORTED_PROVIDERS — all it can write/read remotely
S3_COMPATIBLE_SCHEMES = ("s3", "r2")  # boto3-style credentials; gs uses its own (GOOGLE_APPLICATION_CREDENTIALS)
AWS_CREDENTIAL_ENV_VARS = ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_SHARED_CREDENTIALS_FILE")


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


def sample_to_row(sample: Sample, index: int) -> dict[str, Any]:
    """One decoded sample's non-array fields, for a manifest table row.

    Drops any array/tensor value (the pixel payload) — used by both
    SampleStore.to_parquet and StoreDataset.to_pandas so the two build
    identical rows.

    Args:
        sample: One decoded sample dict.
        index: Row position this sample came from.

    Returns:
        {"index": index, **sample's non-array fields}.
    """
    row: dict[str, Any] = {"index": index}
    row.update({k: v for k, v in sample.items() if not isinstance(v, (np.ndarray, torch.Tensor))})
    return row


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


def warn_if_missing_aws_credentials(path: str, storage_options: dict[str, Any] | None) -> None:
    """Warn if no AWS credentials are visible anywhere boto3 would check.

    Best-effort only — an IAM instance role (EC2/ECS metadata service) works
    without any of these and isn't checked here (would need a real network
    call). A false-positive warning there is the tradeoff for not silently
    waiting until litdata's own read/write fails with an unhelpful error
    several frames deep in a third-party dependency.

    Args:
        path: Resolved s3://.../r2://... path, for the warning message.
        storage_options: As normalize_path is about to return it.
    """
    if storage_options and storage_options.get("aws_access_key_id"):
        return
    if any(os.getenv(var) for var in AWS_CREDENTIAL_ENV_VARS):
        return
    if (Path.home() / ".aws" / "credentials").exists():
        return
    warnings.warn(
        f"{path}: no AWS credentials found in storage_options, "
        f"{'/'.join(AWS_CREDENTIAL_ENV_VARS)}, or ~/.aws/credentials — read/write will "
        "likely fail. Pass storage_options={'aws_access_key_id': ..., 'aws_secret_access_key': ...} "
        "or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."
    )


def normalize_path(
    path: str | Path, storage_options: dict[str, Any] | None
) -> tuple[str | Path, dict[str, Any] | None]:
    """Resolve path to what litdata actually accepts, translating a Storage Bucket URI.

    Dispatches once on scheme — local, hf://buckets/... (Storage Bucket),
    hf://datasets/... (unsupported here), or one of litdata's own s3/gs/r2 —
    then, once, warns if the resolved path is S3-compatible (s3/r2) and no
    AWS credentials look reachable anywhere.

    Args:
        path: As given to SampleStore().
        storage_options: As given in SampleStoreConfig, if any.

    Returns:
        (path litdata should use, storage_options litdata should use).

    Raises:
        ValueError: hf://datasets/... (not writable directly — see the
            upload_dataset_to_hf.py boilerplate script) or any other
            unsupported scheme.
    """
    if not isinstance(path, str) or "://" not in path:
        return path, storage_options  # local path — no scheme, no credentials involved

    scheme = path.split("://", 1)[0]

    if scheme == "hf" and path.startswith(HF_BUCKET_PREFIX):
        # Storage Bucket — rewrite to its equivalent s3:// URI + gateway settings,
        # same AWS-style credentials as any other S3-compatible path
        namespace, s3_path = parse_hf_bucket_path(path)
        resolved_path = s3_path
        resolved_options = {**hf_bucket_storage_options(namespace), **(storage_options or {})}
        warn_if_missing_aws_credentials(resolved_path, resolved_options)
    elif scheme == "hf":
        # hf://datasets/... — a Hub dataset repo, not S3-backed, not writable this way
        raise ValueError(
            f"{path!r} — hf://datasets/... isn't writable directly (litdata only writes "
            f"local or {WRITABLE_SCHEMES}); write there then push with the upload_dataset_to_hf.py "
            f"boilerplate script, or use {HF_BUCKET_PREFIX}<namespace>/<bucket>[/<key>] for a Storage Bucket"
        )
    elif scheme in S3_COMPATIBLE_SCHEMES:
        # s3://, r2:// — used as-is, needs the same AWS-style credentials
        resolved_path, resolved_options = path, storage_options
        warn_if_missing_aws_credentials(resolved_path, resolved_options)
    elif scheme in WRITABLE_SCHEMES:
        # gs:// — litdata's own supported scheme, different credential mechanism
        # (GOOGLE_APPLICATION_CREDENTIALS/ADC), not checked here
        resolved_path, resolved_options = path, storage_options
    else:
        raise ValueError(f"{path!r} — unsupported scheme, must be local or one of {WRITABLE_SCHEMES}")

    return resolved_path, resolved_options
