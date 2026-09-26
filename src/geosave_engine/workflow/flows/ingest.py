"""Prepare a training dataset from a local tree of label rasters."""

from pathlib import Path

from prefect import flow
from pydantic import JsonValue

from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import ModelSpec
from geosave_engine.workflow.tasks import ingest_sample, save_catalog


def _discover_labels(root: Path, pattern: str) -> dict[str, Path]:
    """Return deterministic label identities and local paths."""
    if not root.is_dir():
        raise ValueError(f"Local label root must be a directory: {root}")

    glob = Path(pattern)
    if glob.is_absolute():
        raise ValueError("Label pattern must be relative to the label root")
    if ".." in glob.parts:
        raise ValueError("Label pattern must not contain '..'")

    paths = sorted(
        root.glob(pattern), key=lambda path: path.relative_to(root).as_posix()
    )
    if not paths:
        raise ValueError(f"No labels match {pattern!r} beneath {root}")
    if non_files := [path for path in paths if not path.is_file()]:
        names = [path.relative_to(root).as_posix() for path in non_files]
        raise ValueError(f"Matched label is not a file: {names}")

    labels = {path.relative_to(root).as_posix(): path for path in paths}
    outputs: dict[str, str] = {}
    for sample_id in labels:
        output = Path(sample_id).with_suffix(".zarr").as_posix()
        if previous := outputs.get(output):
            raise ValueError(
                f"Labels {previous!r} and {sample_id!r} map to the same "
                f"sample output {output!r}"
            )
        outputs[output] = sample_id
    return labels


@flow(name="ingest", persist_result=False)
def ingest(
    labels: str,
    sources: dict[str, dict[str, JsonValue]],
    *,
    output: str,
    spec: str,
    pattern: str = "**/*.tif",
) -> str:
    """Prepare matching imagery for every discovered label raster."""
    source_configs = {
        name: SourceConfig.model_validate(source) for name, source in sources.items()
    }
    model = ModelSpec.load(spec)
    if not model.sources:
        raise ValueError("At least one model source is required")
    if "label" in model.sources:
        raise ValueError("Model source name 'label' is reserved for label rasters")

    missing = model.sources.keys() - source_configs.keys()
    extra = source_configs.keys() - model.sources.keys()
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"Source bindings are missing: {sorted(missing)}")
        if extra:
            problems.append(f"Unknown source bindings: {sorted(extra)}")
        raise ValueError("; ".join(problems))

    if "://" in output:
        raise ValueError("Dataset output must be a local directory")
    destination = Path(output)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Dataset output must be a local directory")

    discovered = _discover_labels(Path(labels), pattern)
    pending = {
        sample_id: ingest_sample.submit(
            label,
            source_configs,
            model.sources,
            destination / "samples" / Path(sample_id).with_suffix(".zarr"),
        )
        for sample_id, label in discovered.items()
    }
    completed = {
        sample_id: result.result() for sample_id, result in pending.items()
    }
    return save_catalog.submit(
        completed, destination / "manifest.parquet"
    ).result()
