"""Push a local SampleStore folder to a Hugging Face Hub dataset repo.

Edit constants below, run directly. STORE_PATH must already be written
locally — SampleStore also writes straight to s3://gs://r2:// directly,
this script only covers "write locally, then push to HF".
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token

from geosave_engine.geodata.datastore.sample import SampleStore

STORE_PATH = "data/train"  # local SampleStore folder to push
CHUNK_SIZE = 1000  # must match how STORE_PATH was written — SampleStore needs it even for a read-only open
REPO_ID = "org/dataset-name"  # target HF Hub dataset repo
REPO_TYPE = "dataset"
TOKEN = None  # None reads HF_TOKEN env var, then `hf auth login`'s cached token
CREATE_PARQUET = True  # also push a "<STORE_PATH folder name>.parquet" manifest to the repo root
PATH_IN_REPO = None  # None uploads to repo root


def main() -> None:
    """Push STORE_PATH's chunks (and optionally a parquet manifest) to REPO_ID."""
    token = TOKEN or get_token()
    if token is None:
        raise ValueError(
            "No HF token found (HF_TOKEN env var unset, not logged in via `hf auth login`) "
            "— set TOKEN or log in, unless REPO_ID is a public repo you don't need auth for."
        )
    api = HfApi(token=token)

    if CREATE_PARQUET:
        # manifest built in a temp dir, kept separate from STORE_PATH itself
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / f"{Path(STORE_PATH).name}.parquet"
            SampleStore(STORE_PATH, chunk_size=CHUNK_SIZE).to_parquet(manifest_path)
            api.upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo=manifest_path.name,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )

    api.upload_folder(
        folder_path=STORE_PATH,
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )


if __name__ == "__main__":
    main()
