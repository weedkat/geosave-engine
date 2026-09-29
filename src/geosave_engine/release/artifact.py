"""Publish and load native GeoSave inference releases."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp, TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download

from geosave_engine.__about__ import __version__
from geosave_engine.model_spec import ModelSpec
from geosave_engine.ml.model_chain import ModelChain

_RELEASE_FILES = {
    "README.md",
    "config.json",
    "model.safetensors",
    ModelSpec.filename,
}


def _validated_spec(spec: ModelSpec | str | Path) -> ModelSpec:
    """Load and revalidate one model-owned processing contract."""
    if isinstance(spec, ModelSpec):
        return ModelSpec.model_validate(spec.model_dump())
    loaded = ModelSpec.load(spec)
    return ModelSpec.model_validate(loaded.model_dump())


def _validate_release(model: ModelChain) -> None:
    """Validate that model construction is serializable before writing."""
    json.dumps(model.stage_specs)


def _model_card() -> str:
    """Return the generated release card."""
    return f"""---
library_name: geosave-engine
---

# GeoSave native model

Install the exact release runtime:

```bash
pip install 'geosave-engine[hub]=={__version__}'
```

```python
from geosave_engine.release import load_model, load_spec

spec = load_spec("organization/model", revision="commit-oid")
model = load_model("organization/model", revision="commit-oid")
```
"""


def save_model(
    model: ModelChain,
    path: str | Path,
    *,
    spec: ModelSpec | str | Path,
) -> Path:
    """Write one complete local inference release atomically.

    Args:
        model: Configured native inference graph to persist.
        path: New local release directory.
        spec: Model-owned processing contract or its local path.

    Returns:
        Final release directory.

    Raises:
        FileExistsError: The destination already exists.
        ValueError: The model and processing contract are incompatible.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)

    validated_spec = _validated_spec(spec)
    _validate_release(model)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        from .huggingface import GeoSaveModel

        GeoSaveModel.from_chain(model).save_pretrained(staging)
        validated_spec.save(staging)
        (staging / "README.md").write_text(_model_card(), encoding="utf-8")
        files = {entry.name for entry in staging.iterdir()}
        if files != _RELEASE_FILES:
            raise RuntimeError(
                f"Incomplete model release: expected {sorted(_RELEASE_FILES)}, "
                f"got {sorted(files)}"
            )
        staging.rename(target)
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    return target


def publish_model(
    model: ModelChain,
    repo_id: str,
    *,
    spec: ModelSpec | str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
) -> str:
    """Upload one complete release and return its immutable commit hash.

    Args:
        model: Configured native inference graph to publish.
        repo_id: Hugging Face model repository ID.
        spec: Model-owned processing contract or its local path.
        revision: Target Hub branch or tag.
        token: Hugging Face authentication token or token-selection flag.

    Returns:
        Immutable commit OID created by the upload.
    """
    with TemporaryDirectory() as temporary:
        release = save_model(model, Path(temporary) / "release", spec=spec)
        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            token=token,
        )
        commit = api.upload_folder(
            folder_path=release,
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            token=token,
        )
    return commit.oid


def _local_source(path_or_repo_id: str | Path) -> Path | None:
    """Resolve explicit Paths and existing strings as local sources."""
    if isinstance(path_or_repo_id, Path):
        if not path_or_repo_id.exists():
            raise FileNotFoundError(path_or_repo_id)
        return path_or_repo_id
    path = Path(path_or_repo_id)
    return path if path.exists() else None


def load_model(
    path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> ModelChain:
    """Load one native inference graph from disk or Hugging Face Hub.

    Args:
        path_or_repo_id: Local release path or Hub repository ID.
        revision: Optional Hub commit, tag, or branch.
        token: Hugging Face authentication token or token-selection flag.
        local_files_only: Refuse network downloads when true.

    Returns:
        Native configured model chain.
    """
    from .huggingface import GeoSaveModel

    source = _local_source(path_or_repo_id)
    location = source if source is not None else path_or_repo_id
    if revision is None:
        model = GeoSaveModel.from_pretrained(
            location,
            token=token,
            local_files_only=local_files_only,
        )
    else:
        model = GeoSaveModel.from_pretrained(
            location,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
    return model.chain


def load_spec(
    path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> ModelSpec:
    """Load one processing contract independently of model weights.

    Args:
        path_or_repo_id: Local release path or Hub repository ID.
        revision: Optional Hub commit, tag, or branch.
        token: Hugging Face authentication token or token-selection flag.
        local_files_only: Refuse network downloads when true.

    Returns:
        Validated model-owned processing contract.
    """
    source = _local_source(path_or_repo_id)
    if source is None:
        assert isinstance(path_or_repo_id, str)
        source = Path(
            hf_hub_download(
                repo_id=path_or_repo_id,
                filename=ModelSpec.filename,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
            )
        )
    return ModelSpec.load(source)
